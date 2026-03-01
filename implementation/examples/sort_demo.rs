//! Example: Compile Cole's merge sort to sequential C code.
//!
//! This demonstrates the full PRAM compilation pipeline:
//! 1. Load Cole's merge sort from the algorithm library
//! 2. Generate sequential C code via the codegen pipeline
//! 3. Print the generated code and statistics

use pram_compiler::algorithm_library;
use pram_compiler::codegen::generator::{CodeGenerator, GeneratorConfig};

fn main() {
    println!("=== PRAM Compiler: Cole's Merge Sort Demo ===\n");

    // Load Cole's merge sort from the algorithm library
    let program = algorithm_library::sorting::cole_merge_sort();

    println!("Algorithm: {}", program.name);
    println!("Memory Model: {}", program.memory_model);
    println!("Parameters: {:?}", program.parameters.iter().map(|p| &p.name).collect::<Vec<_>>());
    println!("Shared Memory Regions: {:?}", program.shared_region_names());
    println!("Total Statements: {}", program.total_stmts());
    println!("Parallel Steps: {}", program.parallel_step_count());
    if let Some(ref work) = program.work_bound {
        println!("Work Bound: {}", work);
    }
    if let Some(ref time) = program.time_bound {
        println!("Time Bound: {}", time);
    }
    println!();

    // Generate C code
    let config = GeneratorConfig {
        opt_level: 2,
        include_timing: true,
        include_assertions: true,
        ..GeneratorConfig::default()
    };

    let generator = CodeGenerator::new(config);
    let c_code = generator.generate(&program);

    println!("Generated C code ({} bytes, {} lines):", c_code.len(), c_code.lines().count());
    println!("{:-<60}", "");

    // Print first 80 lines
    for (i, line) in c_code.lines().enumerate() {
        if i >= 80 {
            println!("... ({} more lines)", c_code.lines().count() - 80);
            break;
        }
        println!("{}", line);
    }

    println!("\n{:-<60}", "");
    println!("Compilation complete.");

    // Also demonstrate bitonic sort
    println!("\n=== Bitonic Sort ===\n");
    let bitonic = algorithm_library::sorting::bitonic_sort();
    println!("Algorithm: {}", bitonic.name);
    println!("Memory Model: {}", bitonic.memory_model);
    println!("Total Statements: {}", bitonic.total_stmts());

    let bitonic_c = generator.generate(&bitonic);
    println!("Generated C: {} bytes, {} lines", bitonic_c.len(), bitonic_c.lines().count());
}
