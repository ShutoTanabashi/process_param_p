use process_param::chi2::Parameter;
use process_param::CalcProb;
use process_param::norm::Scenario;

fn main() {
    println!("Hello, world!");
    
    println!("test survival function");
    let v = [-1.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.00, 20.25, 25.0, 30.25];
    let n = Parameter::new(10.0).unwrap();
    for u in v {
        println!("value is {}", n.sf(u));
    }
    print!("\n\n");

    println!("test reading TOML file");
    let path = std::path::Path::new("test/test_scenario.toml");
    let scenario = Scenario::from_toml(&path).unwrap();
    let decomp = scenario.decomplession().unwrap();
    for (i,d) in decomp.iter().enumerate() {
        println!("{} / {:?}", i+1, d);
    }
}
