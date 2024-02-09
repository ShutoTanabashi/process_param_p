//! プロセスを表現する確率分布およびそのパラメータを定義するプログラム
//! 
//! 状態追跡法向けに作成． *(多分抜取検査等でも利用可能です)*

pub mod norm;
pub mod chi2;

extern crate rayon;
use rayon::prelude::*;
extern crate serde;
use serde::{Serialize, Deserialize};
extern crate toml;


// 各種パラメータの型エイリアス
/// 変化点
pub type Tau = u32;
/// 変化回数
pub type NumChg = u32;

/// パラメータ設定に関するエラー
///
/// パラメータの値域が異なる場合などに用いられる．
#[derive(Debug, Clone)]
pub struct ParamError {
    pub message: String,
}

use std::{self, fmt};

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParamError {
    fn description(&self) -> &str {
        &self.message
    }
}

/// シナリオに関するエラー
#[derive(Debug, Clone)]
pub struct ScenarioError {
    pub message: String,
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ScenarioError {
    fn description(&self) -> &str {
        &self.message
    }
}

extern crate rand;
use rand::RngCore;

/// 開区間 $(0,1)$ の一様乱数を生成する
/// 
/// # 引数
/// * `rng` - 乱数生成器
fn open_uni_rg<R: RngCore> ( rng: &mut R ) -> f64 {
    (0.5 + rng.next_u32() as f64) * ( 1.0 / 4_294_967_296.0 )
    // 4,294,967,296 = 2^32
}

/// プロセスを定義
/// 
/// パラメータや取得値の型を定義する．
pub trait Process: Sized {
    /// 取得値の型
    type Observation;
    
    /// パラメータを1つの変数として表現する場合の型
    /// 
    /// # 注意
    /// 利用の際には，この型の時点ではパラメータの値域のチェックは行われておらず，Self型に変換された時点でパラメータとして有効であることが保証される点に注意する．
    /// あくまでもプログラム中に必要な場合のみ利用すること．
    type Param;

    /// パラメータをParam型として取得
    fn param(&self) -> Self::Param;

    /// Param型からSelf型変数を生成
    /// 
    /// # 注意
    /// 特に複数の引数が存在する場合などは，それぞれを独立した引数として受け取ってSelf型変数を返す関数を設定することが望ましい．
    /// その場合，関数名は `new` とすること．
    /// *（引数が異なる場合のtrait関数の宣言方法が自分の技量では分からないので...）*
    fn from_param(param: Self::Param) -> Result<Self, ParamError>;
} 


/// 最尤推定量の計算
pub trait Mle: Process {
    /// データ列からパラメータの最尤推定量を計算しSelfを生成
    fn mle(data: &[Self::Observation]) -> Result<Self, ParamError>;

    /// データ列のスライスからそれぞれの最尤推定量をまとめて計算
    fn mle_all(datas: &[Vec<Self::Observation>]) -> Result<Vec<Self>, ParamError>{
        datas.iter()
             .map(|d| Self::mle(d))
             .collect::<Result<Vec<Self>, ParamError>>()
    }
}

/// プロセスのシミュレーターとして取得値を生成可能
/// 
/// 確率分布を表すパラメータが構造体の要素として格納されていることを想定しています．
pub trait ProcessSimulator: Process {    
    /// プロセスの取得値を表す乱数値を生成
    /// 
    /// # 引数
    /// *  `rng` - 一様乱数生成器
    fn rand<R: RngCore>(&self, rng: &mut R) -> Self::Observation; 

    /// プロセスの取得値をサンプルサイズ n個分生成．
    /// 
    /// # 引数
    /// * `rng` - 一様乱数生成器
    /// * `n` - サンプルサイズ
    fn rand_with_n<R: RngCore>(&self, rng: &mut R, n: usize) -> Vec<Self::Observation> {
        let mut rands = Vec::with_capacity(n);
        for _i in 0..n {
            rands.push(self.rand(rng));
        }
        rands
    }
}

/// 確率点から上側（下側）確率を計算可能
/// 
/// 確率分布を表すパラメータが構造体の要素として格納されていることを想定しています．
pub trait CalcProb: Process {
    /// 確率点から下側確率を計算する
    /// 
    /// # 引数
    /// * `x` - 確率変数
    fn cdf(&self, x: Self::Observation) -> f64;

    /// 確率点から上側確率を計算する
    /// 
    /// # 引数
    /// * `x` - 確率変数
    fn sf(&self, x: Self::Observation) -> f64 {
        1.0 - self.cdf(x)
    }
}



/// 変化パターンを示す列挙型
///
/// # 種類
/// * `Step` - ステップ変化
/// * `Linear` - 線形変化
/// * `StLi` - 線形変化とステップ変化の両方
///
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChangeType<T> {
    /// ステップ変化
    ///
    /// > 要素毎に含まれる構造体要素
    /// > 
    /// > * `level` - パラメータの値
    Step { level: T },
    /// 線形変化
    ///
    /// > 要素毎に含まれる構造体要素
    /// >
    /// > * `grad` - 変化の傾き
    /// > * `init` - 変化発生直前のパラメータの値
    /// >
    /// > **注意**
    /// > `Linear`は本来は傾き `grad` のみがパラメータだが，値を決定するためには初期値 `init`が必要になる．
    /// > そのため `init` も構造体要素に含んでいる．
    /// > 状況によっては `grad` のみを指定する場合が有り得るため， `init` は `Option<T>` で定義．
    Linear {grad: T, init: Option<T>},
    /// 線形変化とステップ変化の混合
    ///
    /// > 要素毎に含まれる構造体要素
    /// > 
    /// > * `grad` - パラメータの変化量．`ax+b`のa
    /// > * `init` - パラメータの初期値．`ax+b`のb
    StLi { grad: T, init: T },
}


impl<T> ChangeType<T> 
where
    T: Clone + Copy + ParamVal + std::ops::Add<Output = T> + std::marker::Send + std::marker::Sync
{
    /// [`ChangeType::Step`]型を示す文字列  
    /// 次の文字列が対応:  
    /// - step
    /// - Step
    /// - S
    pub const KEYWORD_STEP: [&'static str; 3] = ["step", "Step", "S"];
    /// [`ChangeType::Linear`]型を示す文字列  
    /// 次の文字列が対応:  
    /// - linear
    /// - Linear
    /// - L
    pub const KEYWORD_LINEAR: [&'static str; 3] = ["linear", "Linear", "L"];
    /// [`ChangeType::StLi`]型を示す文字列  
    /// 次の文字列が対応:  
    /// - StLi
    /// - stli
    /// - stepandlinear
    /// - StepAndLinear
    /// - SL
    /// - LiSt
    /// - list
    /// - linearandstep
    /// - LinearAndStep
    /// - LS
    pub const KEYWORD_STLI: [&'static str; 10] = ["StLi", "stli", "stepandlinear", "StepAndLinear", "SL", "LiSt", "list", "linearandstep", "LinearAndStep", "LS"];

    /// [`ChangeType::Step`]型を示す文字列であるか判定
    /// 
    /// # 引数
    /// * `change_type` - 変化タイプを示す文字列
    ///
    /// # 使用例
    /// ```
    /// # use process_param::ChangeType;
    /// assert!(ChangeType::<f64>::is_step("S"));
    /// assert!(!ChangeType::<f64>::is_step("LinearAndStep"));
    /// ```
    pub fn is_step(change_type: &str) -> bool {
        Self::KEYWORD_STEP.iter().any(|&k| k==change_type)
    }

    /// [`ChangeType::Linear`]型を示す文字列であるか判定
    /// 
    /// # 引数
    /// * `change_type` - 変化タイプを示す文字列
    ///
    /// # 使用例
    /// ```
    /// # use process_param::ChangeType;
    /// assert!(ChangeType::<f64>::is_linear("L"));
    /// assert!(!ChangeType::<f64>::is_linear("StepAndLinear"));
    /// ```
    pub fn is_linear(change_type: &str) -> bool {
        Self::KEYWORD_LINEAR.iter().any(|&k| k==change_type)
    }

    /// [`ChangeType::StLi`]型を示す文字列であるか判定
    /// 
    /// # 引数
    /// * `change_type` - 変化タイプを示す文字列
    ///
    /// # 使用例
    /// ```
    /// # use process_param::ChangeType;
    /// assert!(ChangeType::<f64>::is_stli("SL"));
    /// assert!(!ChangeType::<f64>::is_stli("step"));
    /// ```
    pub fn is_stli(change_type: &str) -> bool {
        Self::KEYWORD_STLI.iter().any(|&k| k==change_type)
    }


    // スライスの最初の要素を`level`として`Step`を作成
    fn gen_step_from_slice(params: &[T]) -> Result<Self, ParamError> {
        if params.len() < 1 {
            Err(ParamError{
                message: "Unable to create ChangeType instance: Step change needs 1 argment(level).".to_string()
            })
        } else {
            let level = params[0].clone();
            Ok(ChangeType::Step{ level })
        }
    }

    // スライスの最初2個の要素をそれぞれ`grad`と`init`として`Linear`を作成
    // ただし，`init` はなくとも良い．その場合は `Option::None` を代入．
    fn gen_linear_from_slice(params: &[T]) -> Result<Self, ParamError> {
        if params.len() == 0 {
            return Err(ParamError{
                message: "Unable to create ChangeType instance: Linear change needs more than 1 argment(graduation is required, and initial value if available).".to_string()
            })
        }
        let grad = params[0].clone();
        // initがあればその値を利用する．無ければ `Option::None`． 
        let init = if params.len() == 1 {
                None
            } else {
                Some(params[1].clone())
            };
        Ok(ChangeType::Linear{ grad, init })
    }


    /// `ChangeType::Linear` について， `init` が設定されていないかを確認する．
    ///
    /// 設定されていない場合のみ `false` となる．
    /// 設定されている場合および`ChangeType::Linear` 以外の要素の場合に `true` を返す．
    /// 
    /// # 使用例
    /// ```
    /// # use process_param::{ChangeType, ParamVal};
    /// # use process_param::norm::Mu;
    /// // 初期値あり
    /// let param_linear_a = [-0.5, 10.0];
    /// let linear_change_a :ChangeType<Mu> = ChangeType::new("Linear", &param_linear_a).unwrap();
    /// assert!(linear_change_a.linear_has_init());
    /// // 初期値無し
    /// let param_linear_b = [-0.5];
    /// let linear_change_b :ChangeType<Mu> = ChangeType::new("Linear", &param_linear_b).unwrap();
    /// assert!(!linear_change_b.linear_has_init());
    /// // `ChangeType::Linear`以外
    /// let param_step = [2.0];
    /// let step_chage :ChangeType<Mu> = ChangeType::new("Step", &param_step).unwrap();
    /// assert!(step_chage.linear_has_init());
    /// ```
    pub fn linear_has_init(&self) -> bool {
        match self {
            ChangeType::Linear{grad: _, init} => init.is_some(),
            _ => true,
        }
    }


    /// `init` が指定されていない `ChangeType::Linear` に `init` の値を追加する．
    ///
    /// # 引数
    /// * `init_val` - 初期値
    ///
    /// # 使用例
    /// ```
    /// # use process_param::{ChangeType, ParamVal};
    /// # use process_param::norm::Mu;
    /// let param_linear = [-0.5];
    /// let linear_change :ChangeType<Mu> = ChangeType::new("Linear", &param_linear).unwrap();
    /// let linear_change_init = linear_change.add_linear_init(&10.0).unwrap();
    /// assert_eq!(linear_change_init, ChangeType::Linear{ grad: -0.5, init: Some(10.0) });
    /// ```
    ///
    /// 既に値が設定されている場合や， `ChangeType::Linear` 以外の場合はエラーとなる．
    /// ```should_panic
    /// # use process_param::{ChangeType, ParamVal};
    /// # use process_param::norm::Mu;
    /// let param_linear = [-0.5, 10.0];
    /// let linear_change :ChangeType<Mu> = ChangeType::new("Linear", &param_linear).unwrap();
    /// let linear_change_init = linear_change.add_linear_init(&10.0).unwrap();
    /// ```
    pub fn add_linear_init(&self, init_val: &T) -> Result<Self, ParamError> {
        if !self.linear_has_init() {
            let grad = self.get_setting()[0];
            Ok(ChangeType::Linear{grad, init: Some(init_val.clone())})
        } else {
            let message = match self {
                ChangeType::Linear{grad:_, init:_} => "`init` has been set.".to_string(),
                _ => "This variable is not `ChangeType::Linear`.".to_string(),
            };
            Err(ParamError{message})
        }
    }

    // スライスの最初2個の要素をそれぞれ`grad`と`init`として`StLi`を作成
    fn gen_stli_from_slice(params: &[T]) -> Result<Self, ParamError> {
        if params.len() < 2 {
            Err(ParamError{
                message: "Unable to create ChangeType instance: StLi(step and linear) change needs 2 argments(graduation and initial value).".to_string()
            })
        } else {
            let grad = params[0].clone();
            let init = params[1].clone();
            Ok(ChangeType::StLi{ grad, init })
        }
    }


    /// ChangeTypeインスタンスを生成
    ///
    /// # 引数
    /// * `change_type` - 変化パターンの指定．指定に用いる文字列は対応する`KEYWORD_○○`を参照
    /// * `params` - 指定するパラメータの値
    ///
    /// # 注意
    /// `params`については，要素の先頭から順にあてがわれる．
    /// 要素数が多い場合，残りの要素は無視される．
    /// * `Step` 
    ///
    ///     0. `level`  
    ///
    /// * `Linear`
    ///     0. `grad`
    ///     1. `init`
    /// 
    /// * `StLi`  
    ///     
    ///     0. `grad`  
    ///     1. `init`  
    ///
    /// # 使用例
    /// ```
    /// # use process_param::{ChangeType, ParamVal};
    /// # use process_param::norm::Mu;
    /// // ステップ変化
    /// let param_step = vec![10.0];
    /// let step_change :ChangeType<Mu> = ChangeType::new("step", &param_step).unwrap();
    /// assert_eq!(step_change, ChangeType::Step{ level: 10.0 });
    ///
    /// // 線形変化
    /// let param_linear = [-0.5, 10.0];
    /// let linear_change :ChangeType<Mu> = ChangeType::new("Linear", &param_linear).unwrap();
    /// assert_eq!(linear_change, ChangeType::Linear{ grad: -0.5, init: Some(10.0) });
    ///
    /// // 線形&ステップ変化
    /// let param_stli = [2.0, 5.0];
    /// let stli_change :ChangeType<Mu> = ChangeType::new("LS", &param_stli).unwrap();
    /// assert_eq!(stli_change, ChangeType::StLi{ grad: 2.0, init: 5.0 });
    /// ```
    ///
    /// 注意点:パラメータ数が足りない場合にはエラーとなる
    /// ```should_panic
    /// # use process_param::{ChangeType, ParamVal};
    /// # use process_param::norm::Mu;
    /// let param_step: Vec<Mu> = vec![10.0];
    /// let stli_change = ChangeType::new("LS", &param_step).unwrap();
    /// ```
    pub fn new(change_type: &str, params: &[T]) -> Result<Self, ParamError> {
        match change_type {
            ct if Self::is_step(ct) => Self::gen_step_from_slice(params),
            ct if Self::is_linear(ct) => Self::gen_linear_from_slice(params),
            ct if Self::is_stli(ct) => Self::gen_stli_from_slice(params),
            _ => Err(ParamError{
                message: "Unable to create ChangeType instance: Argument `change_type` isn't appropriate.".to_string()
            })
        }
    }



    /// ChangeTypeインスタンスから変化点に定義されたパラメータをVecで取得
    ///
    /// # 使用例
    /// ```
    /// # use process_param::ChangeType;
    /// let stli_change = ChangeType::new("SL", &[2.0, 5.0]).unwrap();
    /// assert_eq!(stli_change.get_setting(), vec![2.0, 5.0])
    /// ```
    pub fn get_setting(&self) -> Vec<T> {
        match self {
            Self::Step{level} => vec![level.clone()],
            Self::Linear{grad, init:_} => vec![grad.clone()],
            // initはパラメータに本来含まれないため省略
            Self::StLi{grad, init} => vec![grad.clone(), init.clone()],
        }
    }


    /// 変化点から任意のタイムステップ経過時のパラメータを取得
    ///
    /// # 引数
    /// * `n` - 変化点からの経過タイムステップ数
    ///
    /// # 注意
    /// この関数にはパラメータの値域の確認はありません．
    /// 必ずParameter構造体などを利用して値域の確認を行ってください．
    ///
    /// また，`ChangeType::Linear`の`init`が`None`の場合には計算不可能であるため，エラーを返します．
    ///
    /// # 使用例
    /// ```
    /// # use process_param::ChangeType;
    /// let stli_change = ChangeType::new("LS", &[2.0, 5.0]).unwrap();
    /// assert_eq!(stli_change.get_param(5).unwrap(), 15.0);
    /// ```
    pub fn get_param(&self, n: Tau) -> Result<T, ParamError>  {
        if !self.linear_has_init() {
            return Err(ParamError{
                message: "`init` of `ChangeType` has not been set.".to_string()
            })
        };
        let val = match self {
            Self::Step{level} => level.clone(),
            Self::Linear{grad, init} => grad.mul_n(n) + init.unwrap().clone(), // initがSomeであることは確認済みであるからunwarp()を使用． 
            Self::StLi{grad, init} => grad.mul_n(n) + init.clone(),
        };
        Ok(val)
    }


    /// 変化点から任意のn期目までのパラメータをVecで取得
    ///
    /// `ChangeType::Linear`の`init`が`None`の場合には計算不可能であるため，エラーを返します．
    ///
    /// # 引数
    /// * `n` - 変化点からの経過タイムステップ数
    ///
    /// # 使用例   
    /// ```
    /// # use process_param::ChangeType;
    /// let stli_change = ChangeType::new("SL", &[2.0, 5.0]).unwrap();
    /// assert_eq!(stli_change.decomplession(5).unwrap(), vec![7.0, 9.0, 11.0, 13.0, 15.0]);
    /// ```
    pub fn decomplession(&self, n: Tau) -> Result<Vec<T>, ParamError> {
       (1..=n).collect::<Vec<Tau>>()
              .par_iter()
              .map(|&n_i| self.get_param(n_i))
              .collect()
    }

    
    /// [`toml::value::Value`]から[`ChangeType::new()`]に必要となる情報を整理したタプルを作成
    ///
    /// # 引数
    /// val_toml - 読み取り対象のTOMLデータ．書式は以下の通り：
    ///
    /// > **example.toml**  
    /// >  
    /// > \# ステップ変化の場合  
    /// > param1 = {type = "Step", level = "0.0"}  
    /// > \# 線形変化の場合
    /// > param2 = {type = "Linear", grad = "0.2", init = "0.0"}
    /// > param3 = {type = "Linear", grad = "0.2"}
    /// > \# 線形+ステップ変化の場合  
    /// > param4 = {type = "Linear", grad = "0.5", init = "1.5"}  
    ///
    /// # 返り値について
    /// 0. 変化タイプを示す文字列
    /// 1. 変化タイプに応じて並び変えた[`toml::value::Value`]．これをパラメータとして適切な型にキャストすれば[`ChangeType::new()`]を適用できる．
    pub fn parse_toml_value(t_val: &toml::value::Value) -> Result<(&str, Vec<&toml::value::Value>), Box<dyn std::error::Error>> {
        let change_type = match t_val.get("type") {
            None => return Err(Box::new(ScenarioError {
                message: "Change type isn't defined at TOML.".to_string(),
            })),
            Some(ct_val) => match ct_val.as_str() {
                None => return Err(Box::new(ScenarioError {
                    message: "Change type is str.".to_string(),
                })),
                Some(ct) => ct,
            },
        };

        let vals = match change_type {
            // ステップ変化
            ct if Self::is_step(ct) => {
                let level = match t_val.get("level") {
                    None => return Err(Box::new(ScenarioError {
                        message: "Step change needs level.".to_string(),
                    })),
                    Some(val_l) => val_l,
                };
                vec![level]
            },
            // 線形変化
            ct if Self::is_linear(ct) => {
                let grad = match t_val.get("grad") {
                    None => return Err(Box::new(ScenarioError {
                        message: "Linear change needs grad.".to_string(),
                    })),
                    Some(val_g) => val_g,
                };
                match t_val.get("init") {
                    None => vec![grad],
                    Some(init) => vec![grad, init],
                }
            }
            // 線形&ステップ変化
            ct if Self::is_stli(ct) => {
                let grad = match t_val.get("grad") {
                    None => return Err(Box::new(ScenarioError {
                        message: "Linear and step change needs grad.".to_string(),
                    })),
                    Some(val_g) => val_g,
                };
                let init = match t_val.get("init") {
                    None => return Err(Box::new(ScenarioError {
                        message: "Linear and step change needs init.".to_string(),
                    })),
                    Some(val_i) => val_i,
                };
                vec![grad, init]
            }
            // `change_type`に該当する変化無し
            _ => return Err(Box::new(ScenarioError {
                message: "Change type doesn't match any change types.".to_string(),
            })),
        };

        Ok((change_type, vals))
    }


    /// TOML形式の文字列に変換
    ///
    /// # 使用例   
    /// ```
    /// # use process_param::ChangeType;
    /// let stli_change = ChangeType::new("SL", &[1.5, 3.5]).unwrap();
    /// let stli_toml = stli_change.to_toml_string();
    /// assert_eq!(stli_toml, r##"{type = "StepAndLinear", grad = 1.5, init = 3.5}"##.to_string());
    /// ```
    pub fn to_toml_string(&self) -> String {
        match self {
            ChangeType::Step{level} => format!("{{type = \"Step\", level = {}}}", level.to_toml_string()),
            ChangeType::Linear{grad, init} => match init {
                    None => format!("{{type = \"Linear\", grad = {}}}", grad.to_toml_string()),
                    Some(init_val) => format!("{{type = \"Linear\", grad = {}, init = {}}}", grad.to_toml_string(), init_val.to_toml_string()),
                }
            ChangeType::StLi{grad, init} => format!("{{type = \"StepAndLinear\", grad = {}, init = {}}}", grad.to_toml_string(), init.to_toml_string()),
        }
    }
}


/// 確率分布のパラメータとして利用できる型
pub trait ParamVal {
    /// パラメータとタイムステップ数([`Tau`]型)の積を計算
    fn mul_n(&self, n: Tau) -> Self;


    /// [`toml`]クレートの[`toml::value::Value`]形式で扱える形式の文字列に変換
    fn to_toml_string(&self) -> String;
}


impl ParamVal for f64 {
    fn mul_n(&self, n: Tau) -> Self {
        self.clone() * (n as f64)
    }

    fn to_toml_string(&self) -> String {
        if self.round() == *self{
            // 小数点以下が0の場合には整数で表示されてしまうため，小数点以下を強制的に表示される
            format!("{:.1}",self)
        } else {
            self.to_string()
        }
    }
}



