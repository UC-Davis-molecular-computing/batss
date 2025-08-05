#![allow(unused_variables)]
use std::collections::HashMap;

use rebop::define_system;
use serde_json::json;
use tqdm::tqdm;

define_system! {
    join_rate split_rate;
    LV { M, D }
    join  : 2 M => D    @ join_rate
    split : D   => 2 M  @ split_rate
}

fn main() {
    println!("Running Lotka-Volterra model with rebop");
    let mut crn = LV::new();
    crn.seed(1);
    let pop_exponent = 3;
    let n = 10isize.pow(pop_exponent);
    crn.join_rate = 1.0 / n as f64;
    crn.split_rate = 1.0;
    crn.M = n;
    crn.D = 0;
    let end_time = 0.5;
    let trials_exponent = 9;
    let trials = 10usize.pow(trials_exponent);
    let mut d_values = Vec::with_capacity(trials);

    let filename = format!(
        "data/dimer_D-counts_time{end_time}_n1e{pop_exponent}_trials1e{trials_exponent}_rebop.json"
    );
    println!(
        "Running rebop directly from Rust collecting dimerization D counts at time {end_time}"
    );
    println!("n=10^{pop_exponent}, trials=10^{trials_exponent}");
    println!("Writing results to {filename}");

    for _ in tqdm(0..trials) {
        crn.M = n;
        crn.D = 0;
        crn.t = 0.0;
        crn.advance_until(end_time);
        d_values.push(crn.D);
    }
    let mut counts = HashMap::new();
    for d in d_values {
        *counts.entry(d).or_insert(0) += 1;
    }
    let json_data = json!(counts);
    // write to file
    // indented with 4 spaces
    let json_data = serde_json::to_string_pretty(&json_data).expect("Failed to serialize JSON");
    std::fs::write(filename, json_data.to_string()).expect("Unable to write file");
}
