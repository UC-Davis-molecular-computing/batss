#![allow(unused_variables)]

use rebop::define_system;
use serde_json::json;

define_system! {
    rabbit_rate fox_rate eat_rate;
    LV { R, F }
    rabbit: R     => 2 R @ rabbit_rate
    fox:    F     =>     @ fox_rate
    eat:    R + F => F   @ eat_rate
}

fn main() {
    println!("Running Lotka-Volterra model with rebop to test runtime sampling.");
    let mut crn = LV::new();
    crn.seed(1);
    let mut elapsed_times = vec![];
    for pop_exponent in 3..=13 {
        println!("Running LV with rebop with n=10^{pop_exponent}");

        let n = 10isize.pow(pop_exponent);
        crn.eat_rate = 1.0 / n as f64;
        crn.rabbit_rate = 1.0;
        crn.fox_rate = 1.0;
        crn.R = n / 2;
        crn.F = n / 2;
        let end_time = 0.5;

        let filename = format!("data/lotka_volterra_time1_times_rebop_rust.json");
        println!("Writing results to {filename}");

        crn.t = 0.0;
        let start_time = std::time::Instant::now();
        crn.advance_until(end_time);
        let elapsed = start_time.elapsed().as_secs_f64();
        elapsed_times.push((n, elapsed));
        println!(
            "{elapsed} seconds to run LV with rebop to time {end_time} with n=10^{pop_exponent}"
        );

        let json_data = json!(elapsed_times);
        // write to file
        // indented with 4 spaces
        let json_data = serde_json::to_string_pretty(&json_data).expect("Failed to serialize JSON");
        std::fs::write(filename, json_data.to_string()).expect("Unable to write file");
    }
}
