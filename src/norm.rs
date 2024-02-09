//! 正規分布に関する定義

use super::{ParamError, ScenarioError, Tau, NumChg, ChangeType};

use std::f64::consts::PI;
use std::path::Path;
use std::fs;
use std::collections::VecDeque;

extern crate serde;
use serde::{Serialize, Deserialize};
extern crate libm;
use libm::{log, sqrt, cos};
extern crate rand;
use rand::RngCore;
extern crate rayon;
use rayon::prelude::*;
extern crate toml;
use toml::value::Array;
extern crate itertools;
// use itertools::Itertools;

// 各種パラメータの型エイリアス
/// 正規分布の平均
pub type Mu = f64;
/// 正規分布の分散
pub type Sigma2 = f64;

use super::open_uni_rg;
/// Box Muller 法に基づいて標準正規乱数を生成する
///
/// # 引数
/// * `rng` - 一様乱数生成器
fn box_muller<R: RngCore> (rng:&mut R) -> f64 {
    // 一様乱数生成
    let u1 = open_uni_rg(rng);
    let u2 = open_uni_rg(rng);

    // tmp
    sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
}


/// Hastings et al.(1995)に基づく標準正規分布の下側確率の計算 
/// 
/// # 引数
/// * `u` - 標準正規分布に従う変数
/// 
/// # 計算式(TeX)
/// > \Phi(u) \fallingdotseq 1 - \frac{1}{2} (1 + d_1 u + d_2 u^2 + d_3 u^3 + d_4 u^4 + d_5 u^5 + d_6 u^6)^{-16}  
/// 
/// ただし，係数 d_i は次の通り: 
///  
/// | d_i | value |  
/// | :-: | :-: |  
/// | d_1 | 0.04986 73470 |  
/// | d_2 | 0.02114 10061 |  
/// | d_3 | 0.00327 76263 |  
/// | d_4 | 0.00003 80036 |  
/// | d_5 | 0.00004 88906 |  
/// | d_6 | 0.00000 53830 |  

fn hastings_cdf(u: f64) -> f64 {
    // 計算式が u >= 0 にて定義されている
    if u < 0.0 {
        1.0 - hastings_cdf(-u)
    } else { 
        1.0 - 0.5 * (HASTINGS_D.iter()
                               .enumerate()
                               .fold(0.0, | acc, (i, d) | acc + d * (u as f64)
                               .powi(i as i32))).powi(-16)
    }
}

const HASTINGS_D: &[f64] = &[1.0, 0.04986_73470, 0.02114_10061, 0.00327_76263, 0.00003_80036, 0.00004_88906, 0.00000_53830];


/// 正規分布に対するパラメータ
///
/// # 引数 
/// * `mu` - 平均 $\mu$  
/// * `sigma2` - 分散 $\sigma^2$  
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    mu: Mu,
    sigma2: Sigma2,
}

impl Parameter {
    /// 新規にParameterを生成  
    ///
    /// # 引数
    /// * `mu` - 平均 $\mu$  
    /// * `sigma2` - 分散 $\sigma^2$  
    /// 
    /// # 使用例
    /// 値が適切ならば`Ok(Parameter)`を返す．  
    /// ```
    /// # use process_param::norm::Parameter;
    /// let pr1 = Parameter::new(0.0, 1.0).unwrap(); 
    /// println!("{:?}", pr1); // Parameter{ mu: 0.0, sigma2: 1.0}
    /// ```
    ///
    /// シナリオとして不適切な値の場合はエラーとなる．  
    /// ```should_panic
    /// # use process_param::norm::Parameter;
    /// let pr2 = Parameter::new(0.0, 0.0).unwrap();  // 分散は正の実数
    /// ```
    pub fn new(mu: Mu, sigma2: Sigma2) -> Result<Self, ParamError> {
        if sigma2 <= 0.0 {
            return Err(ParamError {
                message: r"Variance $\sigma^2$ must be a positive value.".to_string()
            })
        }
        Ok(Parameter{ mu, sigma2 })
    }


    /// 標準正規分布のParameterを生成
    /// 
    /// # 使用例
    /// ```
    /// use process_param::norm::Parameter;
    /// assert_eq!(Parameter::new_standard(), Parameter::new(0.0, 1.0).unwrap());
    /// ```
    pub fn new_standard() -> Self {
        Parameter{ mu: 0.0, sigma2: 1.0}
    }


    /// 確率分布のパラメータ
    /// 正規分布の平均$\mu$を取得．
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// let x = Parameter::new(0.0, 1.0).unwrap();
    /// assert_eq!(x.mu(), 0.0);
    /// ```
    pub fn mu(&self) -> Mu {
        self.mu
    }

    /// 確率分布のパラメータ
    /// 正規分布の分散$\sigma^2$を取得．
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// let x = Parameter::new(0.0, 1.0).unwrap();
    /// assert_eq!(x.sigma2(), 1.0);
    /// ```
    pub fn sigma2(&self) -> Sigma2 {
        self.sigma2
    }
    
    /// 変数を正規化する
    /// 
    /// # 引数
    /// * `x` - 変換対象の引数
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// let x = Parameter::new(1.0, 4.0).unwrap();
    /// assert_eq!(x.normalization(3.0), 1.0);
    /// ```
    pub fn normalization(&self, x: f64) -> f64 {
        (x - self.mu() as f64) / (self.sigma2().sqrt() as f64)
    }
}

use super::Process;
impl Process for Parameter {
    // 取得値の型
    type Observation = f64;
    /// 正規分布のパラメータ
    /// 
    /// 2つのパラメータを(平均，分散)のタプルにて表記する．
    ///    
    /// # 注意
    /// 利用の際には，この型の時点ではパラメータの値域のチェックは行われておらず，Self型に変換された時点でパラメータとして有効であることが保証される点に注意する．
    /// あくまでもプログラム中に必要な場合のみ利用すること．
    type Param = (Mu, Sigma2);

    /// パラメータをParam型として取得
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// # use process_param::Process;
    /// let norm = Parameter::new(0.0, 1.0).unwrap();
    /// assert_eq!(norm.param(), (0.0, 1.0));
    /// assert_eq!(Parameter::from_param(norm.param()).unwrap(), norm);
    /// ```
    fn param(&self) -> Self::Param {
        (self.mu, self.sigma2)
    }

    /// Param型からParameter変数を生成
    /// 
    /// Parameter::new()へのエイリアス．
    fn from_param(param: Self::Param) -> Result<Self, ParamError> {
        Self::new(param.0, param.1)
    }
}

use super::ProcessSimulator;
impl ProcessSimulator for Parameter {
    /// 正規乱数生成器
    ///
    /// # 引数
    /// * `rng` - 一様乱数生成器
    ///
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// # use process_param::{ProcessSimulator};
    /// extern crate rand;
    /// use rand::Rng;
    /// let mut rng = rand::thread_rng();
    /// let param = Parameter::new(0.0, 1.0).unwrap();
    /// let x = param.rand(&mut rng);
    /// println!("{}", x);
    /// ```
    fn rand<R: RngCore>(&self, rng: &mut R) -> Self::Observation {
        (self.mu + self.sigma2.sqrt() * box_muller(rng)) as Self::Observation
    }
}

use super::CalcProb;
impl CalcProb for Parameter {
    /// 確率点から下側確率を計算する
    /// 
    /// # 引数
    /// * `x` - 正規分布に従う変数
    ///  
    /// # 使用例
    /// ```
    /// # use process_param::norm::Parameter;
    /// # use process_param::{CalcProb};
    /// let param = Parameter::new(0.0,1.0).unwrap();
    /// assert_eq!(param.cdf(0.0), 0.5);
    /// ```
    fn cdf(&self, x: Self::Observation) -> f64 {
        hastings_cdf(self.normalization(x as f64))
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
    /// # use process_param::norm::Parameter;
    /// # use process_param::{Mle};
    /// let data = [0.5, 1.5, 0.5, -0.5];
    /// let mle = Parameter::mle(&data).unwrap();
    /// assert_eq!(mle, Parameter::new(0.5, 0.5).unwrap());
    /// ```
    fn mle(data: &[Self::Observation]) -> Result<Self, ParamError> {
        let n = data.len() as Self::Observation;
        let mu = data.iter().fold(0.0, |acc, x| acc + x) / n;
        // Observation型が変更された場合に対応できるように，powメソッドは利用せず
        let sigma2 = data.iter().fold(0.0, |acc, x| acc + (x - mu)*(x - mu)) / n;
        Self::from_param((mu, sigma2))
    }
}


/// 正規分布に対するシナリオ構成要素
///
/// # 引数
/// * `tau` - 変化点 $\tau$  
/// * `mu` - 平均 $\mu$ の変化パターン 
/// * `sigma2` - 分散 $\sigma^2$ の変化パターン
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChangePoint {
    tau: Tau,
    mu: ChangeType<Mu>,
    sigma2: ChangeType<Sigma2>,
}

impl ChangePoint {
    /// 新規にChangePointを生成  
    ///
    /// # 引数
    /// * `tau` - 変化点 $\tau$  
    /// * `mu` - 平均 $\mu$  
    /// * `sigma2` - 分散 $\sigma^2$  
    ///
    /// # 使用例 
    /// ```
    /// # use process_param::ChangeType;
    /// # use process_param::norm::ChangePoint;
    /// let mu = ChangeType::new("Step", &[0.0]).unwrap();
    /// let sigma2 = ChangeType::new("LiSt", &[1.0, 1.5]).unwrap();
    /// let cp1 = ChangePoint::new( 10, mu, sigma2 ); 
    /// println!("{:?}", cp1)
    /// ```
    pub fn new(tau: Tau, mu: ChangeType<Mu>, sigma2: ChangeType<Sigma2>) -> ChangePoint {
        ChangePoint { tau, mu, sigma2 }
    }

    /// 変化点 $\tau$の値
    pub fn tau(&self) -> Tau {
        self.tau
    }

    /// 平均 $\mu$ の変化
    pub fn mu(&self) -> ChangeType<Mu> {
        self.mu
    }

    /// 分散$\sigma^2$ の変化
    pub fn sigma2(&self) -> ChangeType<Sigma2> {
        self.sigma2
    }

    /// 変化点から任意のタイムステップ経過時のパラメータを取得
    ///
    /// # 引数
    /// * `n` - 変化点からの経過タイムステップ数
    ///
    /// # 使用例 
    /// ```
    /// # use process_param::ChangeType;
    /// # use process_param::norm::{ChangePoint, Parameter};
    /// let mu = ChangeType::new("Step", &[0.0]).unwrap();
    /// let sigma2 = ChangeType::new("LiSt", &[1.0, 1.5]).unwrap();
    /// let cp1 = ChangePoint::new( 10, mu, sigma2 ); 
    /// let cp1_5 = cp1.get_param(5).unwrap();
    /// assert_eq!(cp1_5, Parameter::new(0.0, 6.5).unwrap());
    /// ```
    /// 
    /// パラメータが不適切な値となる場合はErrorとなる．
    /// ```should_panic
    /// # use process_param::ChangeType;
    /// # use process_param::norm::ChangePoint;
    /// let mu = ChangeType::new("Step", &[0.0]).unwrap();
    /// let sigma2 = ChangeType::new("LiSt", &[-1.0, 1.5]).unwrap();
    /// let cp2 = ChangePoint::new( 15, mu, sigma2 ); 
    /// let cp2_5 = cp2.get_param(5).unwrap(); // panic
    /// ```
    pub fn get_param(&self, n: Tau) -> Result<Parameter, ParamError> {
        let mu = self.mu().get_param(n)?;
        let sigma2 = self.sigma2().get_param(n)?;
        Parameter::new(mu, sigma2)
    }


    /// 変化点から任意のn期目までのパラメータをVecで取得
    ///
    /// # 引数
    /// * `n` - 変化点からの経過タイムステップ数
    ///
    /// # 使用例 
    /// ```
    /// # use process_param::ChangeType;
    /// # use process_param::norm::{ChangePoint, Parameter};
    /// let mu = ChangeType::new("Step", &[0.0]).unwrap();
    /// let sigma2 = ChangeType::new("LiSt", &[1.0, 1.5]).unwrap();
    /// let cp1 = ChangePoint::new( 10, mu, sigma2 ); 
    /// let decomp1 = cp1.decomplession(3).unwrap();
    /// let ans = vec![
    ///               Parameter::new(0.0, 2.5).unwrap(),
    ///               Parameter::new(0.0, 3.5).unwrap(),
    ///               Parameter::new(0.0, 4.5).unwrap()
    ///           ];
    /// assert_eq!(decomp1, ans);
    /// ```
    /// 
    /// パラメータが不適切な値となる場合はErrorとなる．
    /// ```should_panic
    /// # use process_param::ChangeType;
    /// # use process_param::norm::ChangePoint;
    /// let mu = ChangeType::new("Step", &[0.0]).unwrap();
    /// let sigma2 = ChangeType::new("LiSt", &[-1.0, 1.5]).unwrap();
    /// let cp2 = ChangePoint::new( 15, mu, sigma2 ); 
    /// let decomp2 = cp2.decomplession(3).unwrap(); // panic
    /// ```
    pub fn decomplession(&self, n: Tau) -> Result<DecompParam, ParamError> {
        (1..=n).collect::<Vec<Tau>>()
               .par_iter()
               .map(|&n_i| self.get_param(n_i))
               .collect()
    }


    /// TOML形式の文字列に変換
    pub fn to_toml_string(&self) -> String {
        format!("{{tau = {}, mu = {}, sigma2 = {}}}", self.tau(), self.mu().to_toml_string(), self.sigma2().to_toml_string())
    }
}


/// 正規分布に基づく乱数生成シナリオ
/// 
/// # 引数
/// * `n` - サンプルサイズ $n$
/// * `parameters` - プロセスのパラメータ変化を記述したシナリオ．変化点とその時点までの正規分布のパラメータが記載される．
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    n: Tau,
    parameters: Vec<ChangePoint>,
}

// TOML形式のテンプレートファイルを読み取るための構造体
// プログラム内で利用するシナリオ(Scenario)とは若干形式が異なるため別で定義
#[derive(Serialize, Deserialize, Debug)]
struct ScenarioToml {
    n: isize,
    parameter: Array,
}

/// シナリオに含まれるパラメータを1期間毎に展開
pub type DecompParam = Vec::<Parameter>;


impl Scenario {
    /// 新規にScenarioを生成
    /// 
    /// # 引数
    /// * `n` - サンプルサイズ $n$
    /// * `params` - プロセスのパラメータ変化を記述したシナリオ．
    /// 
    /// # 注意
    /// 引数 `param` はTauの昇順になっている必用がある．
    pub fn new(n: Tau, params: &[(Tau, ChangeType<Mu>, ChangeType<Sigma2>)]) -> Result<Self, Box<dyn std::error::Error>> {
        // サンプルサイズの確認
        if n == 0 {
            return Err(Box::new(ScenarioError {
                message: r"The sample size n must be greater than 0.".to_string()
            }))
        };
        
        // パラメータの初期値が不定になっていないか確認
        // 変数は後に使いまわすためにmutとする
        let (mut last_tau, mut last_mu, mut last_sig2) = params[0];
        for p in [last_mu, last_sig2] {
            if !p.linear_has_init() {
                return Err(Box::new(ScenarioError{
                    message: "The initial value has not set. Please add `init` parameter to `ChangeType::Lienar`.".to_owned()
                }))
            }
        };

        let mut parameters: Vec<ChangePoint> = Vec::with_capacity(params.len());
        // 最初の状態を保存
        parameters.push(ChangePoint::new(last_tau, last_mu, last_sig2));
        let mut prev_gap_tau = 1; 

        // 2個目以降の変化点を順番に追加
        for p in &params[1..] {
            // tauの順序確認
            if p.0 < last_tau {
                return Err(Box::new(ScenarioError {
                    message: "Scenario is not lined up in the order of tau.".to_string()
                }))
            } 
            let p_tau = p.0;
            // 線形変化の確認
            // 初期値がない場合には前の状態の値を利用する
            let p_mu = if p.1.linear_has_init() {
                p.1
            } else {
                let init_val = last_mu.get_param(prev_gap_tau)?;
                p.1.add_linear_init(&init_val)?
            };
            let p_sig2 = if p.2.linear_has_init() {
                p.2
            } else {
                let init_val = last_sig2.get_param(prev_gap_tau)?;
                p.2.add_linear_init(&init_val)?
            };
            // 値の更新
            // let param_line = ChangePoint::new(p_tau, p_mu, p_sig2);
            parameters.push(ChangePoint::new(p_tau, p_mu, p_sig2));
            prev_gap_tau = p_tau - last_tau;
            (last_tau, last_mu, last_sig2) = (p_tau, p_mu, p_sig2);
        }
        return Ok(Scenario{n, parameters})
    }

    /// TOMLからシナリオ読み取り
    ///
    /// # 引数
    /// * `path` - 読み取りたいTOMLファイル
    ///     * *注意* TOMLファイルの中身は以下の通り．
    /// 
    /// > **example.toml**  
    /// >  
    /// > \# サンプル・サイズ  
    /// > n = 10  
    /// >   
    /// > \# パラメータの変化  
    /// > parameter = [  
    /// >     \# それぞれ変化点，平均，分散．変化点に並べる．  
    /// >     {tau = 15, mu = {type = "Step", level = 0.0}, sigma2 = {type = "Step", level = 1.0}},  
    /// >     {tau = 40, mu = {type = "Step", level = 1.0}, sigma2 = {type = "LinearAndStep", grad = 0.5, init = 1.5}}  
    /// > ]  
    ///
    /// # 使用例
    /// ```
    /// # use process_param::norm::Scenario;
    /// let path = std::path::Path::new("test/test_scenario.toml");
    /// let scenario = Scenario::from_toml(&path).unwrap();
    /// println!("{:?}", scenario);
    /// ```
    pub fn from_toml<P: AsRef<Path>>(path: &P) -> Result<Self, Box<dyn std::error::Error>> {
        let file_str = fs::read_to_string(path)?;
        Self::parse_toml_str(&file_str)
    }

    /// TOML形式の文字列からシナリオ読み取り
    ///
    /// # 引数
    /// * `toml_str` - 読み取りたいTOML形式の文字列
    pub fn parse_toml_str(toml_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file_toml: ScenarioToml = toml::from_str(toml_str)?; 
        let mut parameters = Vec::new();
        for p in &file_toml.parameter {
            let tau = match p.get("tau") {
                None => return Err(Box::new(ScenarioError {
                    message: "Scenario needs tau.".to_string(),
                })),
                Some(val_t) => match val_t.as_integer() {
                    None => return Err(Box::new(ScenarioError {
                        message: "Tau is integer.".to_string(),
                    })),
                    Some(t) => t as Tau,
                }
            };
            let mu = match p.get("mu") {
                None => return Err(Box::new(ScenarioError {
                    message: "Scenario needs mu.".to_string(),
                })),
                Some(m_val) => {
                    let (change_type, params_val) = ChangeType::<Mu>::parse_toml_value(m_val)?;
                    let op_params = params_val.iter()
                                              .map(|&val| val.as_float())
                                              .collect::<Option<Vec<f64>>>();
                    let params = match op_params {
                        None => return Err(Box::new(ScenarioError {
                            message: "Parameter mu is float.".to_string(),
                        })),
                        Some(p) => p, 
                    };
                    ChangeType::new(change_type, &params)?
                }
            };
            let sigma2 = match p.get("sigma2") {
                None => return Err(Box::new(ScenarioError {
                    message: "Scenario needs sigma2.".to_string(),
                })),
                Some(s_val) => {
                    let (change_type, params_val) = ChangeType::<Sigma2>::parse_toml_value(s_val)?;
                    let op_params = params_val.iter()
                                              .map(|&val| val.as_float())
                                              .collect::<Option<Vec<f64>>>();
                    let params = match op_params {
                        None => return Err(Box::new(ScenarioError {
                            message: "Parameter sigma2 is float.".to_string(),
                        })),
                        Some(p) => p, 
                    };
                    ChangeType::new(change_type, &params)?
                }    
            };
            parameters.push((tau, mu, sigma2));
        }
        // 返り値が異なるためため一度変数へ格納する．
        let scenario = Self::new(file_toml.n as Tau, &parameters)?;
        Ok(scenario)
    }


    /// TOML形式の文字列に変換
    ///
    /// # 注意
    /// 文字列を上手く結合できない場合panicとなる事がある．
    pub fn to_toml_string(&self) -> String {
        let before_param = format!("n = {}\nparameter = [\n\t", self.n as usize);
        let params = self.parameters()
                         .par_iter()
                         .map(|p| p.to_toml_string())
                         .collect::<Vec<String>>()
                         .join(",\n\t");
        before_param + &params + "\n]"

    }


    /// ScenarioからDecompParamを作成する
    pub fn decomplession (&self) -> Result<DecompParam, ScenarioError> {
        let params = self.parameters();
        
        // 最後の変化点
        let last_param;
        match params.last() {
            Some(l) => last_param = l,
            None => return Err(ScenarioError{
                    message: "This scenario has no change points.".to_string()
                })
        }

        let mut decomp = Vec::with_capacity(last_param.tau() as usize);
        let mut last_tau = 0;
        
        for cp in self.parameters() {
            let n = cp.tau() - last_tau;
            last_tau = cp.tau();
            match cp.decomplession(n) {
                Ok(mut dec_cp) => decomp.append(&mut dec_cp),
                Err(_) => return Err(ScenarioError{
                    message: "This scenario does not satisfy the range of parameters.".to_string()
                })
            }
        }
        Ok(decomp)
    }


    /// Senarioから，最後の変化点以外の[`DecompParam`]を作成する
    ///
    /// 管理図が管理外れ状態を検出するまで乱数を生成する際に用いることを想定．
    /// Scenarioに変化点が1個しかない場合，[`DecompParam`]は要素数0の[`Vec`]となる．
    ///
    /// # 返り値
    /// * `(incontrol, decomp, last_param)` 
    ///     * `incontrol` - 管理状態でのパラメータによる[`DecompParam`]
    ///     * `decomp` - 最後の変化点以外の管理外れ状態でのパラメータによる[`DecompParam`]
    ///     * `last_param` - 最後の変化点
    pub fn decomp_exclude_last(&self) -> Result<(DecompParam, DecompParam, ChangePoint), ScenarioError> {
        let mut params = VecDeque::from_iter(self.parameters().iter());
        
        // 最後の変化点
        let last_param;
        match params.pop_back() {
            Some(l) => last_param = l,
            None => return Err(ScenarioError{
                    message: "This scenario has no change points.".to_string()
                })
        }

        // 最初の変化点
        let incontrol;
        match params.pop_front() {
            Some(f) => incontrol = match f.decomplession(f.tau()) {
                                    Err(e) => return Err(ScenarioError{
                                        message: e.message
                                    }),
                                    Ok(val) => val,
                                    },
            None => return Err(ScenarioError{
                message: "This scenario has only one change point.".to_string()
                }),
        }

        
        // その他の変化点
        let decomp = match params.back() {
            None => Vec::new(),
            Some(l) => {
                let mut decomp_cps = Vec::with_capacity(l.tau() as usize);
                let mut last_tau = self.first_changepoint();
                for cp in &params {
                    let n = cp.tau() - last_tau;
                    last_tau = cp.tau();
                    match cp.decomplession(n) {
                        Ok(mut dec_cp) => decomp_cps.append(&mut dec_cp),
                        Err(_) => return Err(ScenarioError{
                            message: "This scenario does not satisfy the range of parameters.".to_string()
                        })
                    }
                }
                decomp_cps
            }
        };

        Ok((incontrol, decomp, last_param.clone()))
    }

    /// シナリオに記載されたサンプル数 $n$ 
    pub fn n(&self) -> Tau {
        self.n
    }

    /// シナリオに記載されたサンプル数 $n$ を[`usize`]型で取得
    pub fn n_as_usize(&self) -> Result<usize, ScenarioError> {
        match usize::try_from(self.n()){
            Ok(val) => Ok(val),
            Err(_) => return Err(ScenarioError{
                message: "Sample size n doesn't convert to usize.".to_string()
            }),
        }
    }

    /// 変化点数を取得
    pub fn num_change(&self) -> NumChg {
        (self.parameters.len() - 1) as NumChg
    }

    /// パラメータ群の参照
    pub fn parameters(&self) -> &[ChangePoint] {
        &self.parameters
    }

    /// 管理状態のパラメータ
    /// 
    /// # 返り値
    /// * `(mu_0, sigma_0_2)` - それぞれ平均，分散
    pub fn param_in_control(&self) -> (Mu, Sigma2) {
        let incontrol = self.parameters()[0];
        let mu_0 = incontrol.mu().get_param(1).unwrap();
        let sigma_0_2 = incontrol.sigma2().get_param(1).unwrap();
        (mu_0, sigma_0_2)
    }

    /// サンプル平均が従う分布のパラメータ
    ///
    /// 正規分布の再生性より，サンプル平均も正規分布に従う
    ///
    /// # 返り値
    /// * `(mu_barx, sigma_barx_2)`
    ///     * `mu_barx` - サンプル平均の平均．サンプル自体の平均と一致する．
    ///     * `sigma_barx_2` - サンプル平均の分散．サンプル数に応じて減少する．
    pub fn param_samplemean(&self) -> (Mu, Sigma2) {
        let (mu_barx, sigma_0_2) = self.param_in_control();
        let n = self.n() as f64;
        let sigma_barx_2 = sigma_0_2 / n;
        (mu_barx, sigma_barx_2)
    }
    
    /// シナリオに設定された最初の変化点
    pub fn first_changepoint(&self) -> Tau {
        let incontrol = self.parameters()[0];
        incontrol.tau()
    }


    /// \bar{X}管理図の管理限界
    ///
    /// JIS規格(JIS Z9020-2:2016)における「標準値を与えている場合の管理限界」に該当．
    /// 管理限界自体は3シグマ法に従い計算
    /// 
    /// # 計算式
    /// UCL = \mu_0 + 3 \sigma_0 / \sqrt{n},
    /// LCL = \mu_0 - 3 \sigma_0 / \sqrt{n},
    ///
    /// # 返り値
    /// * `(lcl, ucl)` - それぞれ下側管理限界，上側管理限界
    pub fn control_limit_xbar(&self) -> (<Parameter as Process>::Observation, <Parameter as Process>::Observation) {
        let (mu_0, sigma_0_2) = self.param_in_control();
        let n = self.n();
        let ucl = mu_0 + (3.0 * sigma_0_2.sqrt() / ((n as <Parameter as Process>::Observation).sqrt()));
        let lcl = mu_0 - (3.0 * sigma_0_2.sqrt() / ((n as <Parameter as Process>::Observation).sqrt()));
        (lcl, ucl)
    }


    /// s管理図の管理限界
    ///
    /// JIS規格(JIS Z9020-2:2016)における「標準値を与えている場合の管理限界」に該当．
    ///
    /// # 計算式
    /// LCL = \sigma_0 B_5
    /// UCL = \sigma_0 B_6
    /// B_5 = \max(c_4 - 3 \sqrt{1 - c_4^2}, 0)
    /// B_6 = c_4 + 3 \sqrt{1 - c_4^2}
    /// c_4 = \sqrt{2 / (n-1)} \Gamma(n/2) / \Gamma((n-1)/2)
    ///
    /// # 返り値
    /// * `(lcl, ucl)` - それぞれ下側管理限界，上側管理限界
    pub fn control_limit_s(&self) -> (<Parameter as Process>::Observation, <Parameter as Process>::Observation) {
        let (_, sigma_0_2) = self.param_in_control();
        let sigma_0 = sigma_0_2.sqrt();
        let n = self.n() as <Parameter as Process>::Observation;
        let c_4 = (2.0 / (n - 1.0)).sqrt() * libm::tgamma(n / 2.0) / libm::tgamma((n - 1.0) / 2.0);
        let coef = 3.0 * (1.0 - c_4.powi(2)).sqrt(); 
        let ucl = sigma_0 * (c_4 + coef);
        let lcl = sigma_0 * (c_4 - coef).max(0.0);
        (lcl, ucl)
    }


    // パラメータが管理限界を超えているか確認
    //
    // # 引数
    // * `lcl_xbar` - \bar{X}管理図の下側管理限界
    // * `ucl_xbar` - \bar{X}管理図の上側管理限界
    // * `lcl_s` - s管理図の下側管理限界
    // * `ucl_s` - s管理図の上側管理限界
    // * `param` - 判定対象のパラメータ
    fn judge_control_limit(lcl_xbar: &<Parameter as Process>::Observation,
                           ucl_xbar: &<Parameter as Process>::Observation,
                           lcl_s: &<Parameter as Process>::Observation,
                           ucl_s: &<Parameter as Process>::Observation,
                           param: &Parameter) -> bool {
        let (mu, sigma_2) = param.param();
        let sigma = sigma_2.sqrt();
        if mu < *lcl_xbar || mu > *ucl_xbar || sigma < *lcl_s || sigma > *ucl_s {
            false
        } else {
            true
        }
    }


    /// パラメータが管理状態にあるか確認
    /// 
    /// 管理状態であれば[`true`]を返す
    ///
    /// # 引数
    /// * `param` - 判定対象のパラメータ
    pub fn in_control(&self, param: &Parameter) -> bool {
        let (lcl_xbar, ucl_xbar) = self.control_limit_xbar();
        let (lcl_s, ucl_s) = self.control_limit_s();
        Self::judge_control_limit(&lcl_xbar, &ucl_xbar, &lcl_s, &ucl_s, param)
    }


    /// パラメータが管理外れ状態にあるか確認
    ///
    /// 管理外れ状態であれば[`true`]を返す
    ///
    /// # 引数
    /// * `param` - 判定対象のパラメータ
    pub fn out_of_control(&self, param: &Parameter) -> bool {
        !self.in_control(param)
    }


    /// すべてのパラメータが管理状態にあるか確認
    ///
    /// 管理状態であれば[`true`]を返す
    ///
    /// # 引数
    /// * `params` - 判定対象のパラメータ
    pub fn in_control_all(&self, params: &[Parameter]) -> bool {
        let (lcl_xbar, ucl_xbar) = self.control_limit_xbar();
        let (lcl_s, ucl_s) = self.control_limit_s();
        params.par_iter()
              .all(|param| Self::judge_control_limit(&lcl_xbar, &ucl_xbar, &lcl_s, &ucl_s, param))
    }


    /// Parameterのスライスから管理限界を超える時点を見つける
    ///
    /// 管理限界を超えない場合は[`None`]を返す．
    ///
    /// # 引数
    /// * `params` - 判定対象のパラメータ
    pub fn index_out_of_control(&self, params: &[Parameter]) -> Option<usize> {
        let (lcl_xbar, ucl_xbar) = self.control_limit_xbar();
        let (lcl_s, ucl_s) = self.control_limit_s();
        params.par_iter()
              .position_first(|param| !Self::judge_control_limit(&lcl_xbar, &ucl_xbar, &lcl_s, &ucl_s, param))
    }
}

