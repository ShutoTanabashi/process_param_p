//! カイ二乗分布に関する定義

extern crate serde;
use serde::{Serialize, Deserialize};

use super::ParamError;

// 各種パラメータの型エイリアス
/// 強度パラメータ
pub type Lambda = f64;


use super::norm;
/// Wilson-Hilferty(1931)に基づくカイ二乗分布の下側確率の計算
/// 
/// # 引数
/// * `x` - カイ二乗分布に従う変数
/// * `lambda` - カイ二乗分布の強度
/// 
/// # 計算式(TeX)
/// 自由度 $\nu$ のカイ二乗分布の場合，変数 $\chi^2$に対して次の計算を行う:
/// 
/// > P(\chi^2; \nu) \fallingdotseq \Phi(u)
/// 
/// ここで関数 $\Phi(z)$ は標準正規分布の変数 $z$ に対する下側確率ある．
/// さらに $u$ は以下の式で計算される．
/// 
/// > u = \frac{(\chi^2 / \nu)^{1/3} - (1 - (2 / (9 \nu)))}{\sqrt{2 / (9 \nu)}}
fn wh_cdf(x: f64, lambda: Lambda) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        let l = 2.0 / (9.0 * lambda) as f64;
        let u = ((x as f64 / lambda as f64).powf(1.0/3.0) - 1.0 + l) / l.sqrt();
        // let u = (2.0 * x).sqrt() - (2.0 * lambda - 1.0).sqrt();
        norm::Parameter::new_standard().cdf(u)
    }
}

/// カイ二乗分布に対するパラメータ
/// 
/// # 引数
/// * `lambda` - 強度 $\lambda$
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    lambda: Lambda,
}

impl Parameter {
    /// 新規にParameterを生成  
    ///
    /// # 引数
    /// * `lambda` - 強度 $\lambda$
    /// 
    /// # 使用例
    /// 値が適切ならば`Ok(Parameter)`を返す．  
    /// ```
    /// # use process_param::chi2::Parameter;
    /// let pr1 = Parameter::new(2.0).unwrap(); 
    /// println!("{:?}", pr1); // Parameter{ lambda: 2.0 }
    /// ```
    ///
    /// シナリオとして不適切な値の場合はエラーとなる．  
    /// ```should_panic
    /// # use process_param::chi2::Parameter;
    /// let pr2 = Parameter::new(-1.0).unwrap();  // 強度は正の実数
    /// ```
    pub fn new(lambda: Lambda) -> Result<Self, ParamError> {
        if lambda <= 0.0 {
            return Err(ParamError {
                message: r"Rate parameter $\lambda$ must be a positive value.".to_string()
            })
        }
        Ok(Parameter{ lambda })
    }


    /// 確率分布のパラメータ
    /// カイ二乗分布の強度$\lambda$を取得．
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::chi2::Parameter;
    /// let x = Parameter::new(2.0).unwrap();
    /// assert_eq!(x.lambda(), 2.0);
    /// ```
    pub fn lambda(&self) -> Lambda {
        self.lambda
    }
}

use super::CalcProb;
impl CalcProb for Parameter {
    /// 確率点から下側確率を計算する
    /// 
    /// # 引数
    /// * `x` - カイ二乗分布に従う変数
    ///  
    /// # 使用例
    /// ```
    /// # use process_param::chi2::Parameter;
    /// # use process_param::{CalcProb};
    /// let param = Parameter::new(10.0).unwrap();
    /// println!("{}", param.cdf(9.0)); // 約0.46790
    /// ```
    fn cdf(&self, x: Self::Observation) -> f64 {
        wh_cdf(x as f64, self.lambda())
    }
}

use super::Process;
impl Process for Parameter {
    // 取得値の型
    type Observation = f64;
    
    /// パラメータ
    /// 
    /// ポアソン分布は強度パラメータのみで特徴づけられる．
    /// 
    /// # 注意
    /// 利用の際には，この型の時点ではパラメータの値域のチェックは行われておらず，Self型に変換された時点でパラメータとして有効であることが保証される点に注意する．
    /// あくまでもプログラム中に必要な場合のみ利用すること．
    type Param = Lambda;

    /// パラメータを取得
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::chi2::Parameter;
    /// # use process_param::Process;
    /// let poisson = Parameter::new(5.0).unwrap();
    /// assert_eq!(poisson.param(), 5.0);
    /// ```
    fn param(&self) -> Self::Param {
        self.lambda
    }

    /// Param型からParameterを生成
    /// 
    /// Parameter::new()へのエイリアス．
    fn from_param(param: Self::Param) -> Result<Self, ParamError> {
        Self::new(param)
    }
}

use super::Mle;
impl Mle for Parameter {
        /// データ列から正規分布の最尤推定量を計算しParamterを生成
    /// 
    /// # 引数
    /// * `data` - 最尤推定量を推定する対象であるデータ列
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::chi2::Parameter;
    /// # use process_param::{Mle};
    /// let data = [1.0, 3.0, 5.0];
    /// let mle = Parameter::mle(&data).unwrap();
    /// assert_eq!(mle, Parameter::new(3.0).unwrap());
    /// ```
    /// 
    /// ポアソン変数の値域に負の値は含まれないため，引数の要素内に負の値があればエラーを返す．
    /// ```should_panic
    /// # use process_param::chi2::Parameter;
    /// # use process_param::{Mle};
    /// let data = [1.0, 3.0, -5.0];
    /// let mle = Parameter::mle(&data).unwrap(); // 最後の要素が不適切
    /// ```
    fn mle(data: &[Self::Observation]) -> Result<Self, ParamError> {
        // ポアソン変数の値域の確認
        if data.iter().any(|&x| x < 0.0) {
            return Err(ParamError {
                message: r"Poisson variable $x$ must be a positive value.".to_string()
            })
        }
        
        // 最尤推定量計算
        let n = data.len() as Self::Observation;
        let lambda = data.iter().fold(0.0, |acc, x| acc + x) / n;
        Self::from_param(lambda)
    }
}