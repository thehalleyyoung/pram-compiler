//! Example: Compile Shiloach-Vishkin connected components to sequential C code.
//!
//! Demonstrates compiling a graph algorithm from PRAM IR to sequential C,
//! including the hash-partition analysis and Brent scheduling steps.

use pram_compiler::algorithm_library;
use pram_compiler::codegen::generator::{CodeGenerator, GeneratorConfig};

fn main() {
    println!("=== PRAM Compiler: Connectivity Algorithms Demo ===\n");

    // --- Shiloach-Vishkin Connected Components ---
    let sv = algorithm_library::graph::shiloach_vishkin();
    println!("Algorithm: {}", sv.name);
    println!("Memory Model: {}", sv.memory_model);
    println!("Total Statements: {}", sv.total_stmts());
    println!("Parallel Steps: {}", sv.parallel_step_count());
    if let Some(ref desc) = sv.description {
        println!("Description: {}", desc);
    }
    if let Some(ref work) = sv.work_bound {
        println!("Work Bound: {}", work);
    }
    if let Some(ref time) = sv.time_bound {
        println!("Time Bound: {}", time);
    }
    println!();

    let config = GeneratorConfig {
        opt_level: 2,
        include_timing: true,
        include_assertions: false,
        ..GeneratorConfig::default()
    };

    let generator = CodeGenerator::new(config);
    let c_code = generator.generate(&sv);

    println!("Generated C code ({} bytes, {} lines):", c_code.len(), c_code.lines().count());
    println!("{:-<60}", "");

    for (i, line) in c_code.lines().enumerate() {
        if i >= 60 {
            println!("... ({} more lines)", c_code.lines().count() - 60);
            break;
        }
        println!("{}", line);
    }

    // --- Vishkin's Deterministic Connectivity ---
    println!("\n\n=== Vishkin's Deterministic Connectivity ===\n");
    let vishkin = algorithm_library::connectivity::vishkin_connectivity();
    println!("Algorithm: {}", vishkin.name);
    println!("Memory Model: {}", vishkin.memory_model);
    println!("Total Statements: {}", vishkin.total_stmts());

    let vishkin_c = generator.generate(&vishkin);
    println!("Generated C: {} bytes, {} lines", vishkin_c.len(), vishkin_c.lines().count());

    // --- Ear Decomposition ---
    println!("\n=== Ear Decomposition (Reif) ===\n");
    let ear = algorithm_library::connectivity::ear_decomposition();
    println!("Algorithm: {}", ear.name);
    println!("Memory Model: {}", ear.memory_model);
    println!("Total Statements: {}", ear.total_stmts());

    let ear_c = generator.generate(&ear);
    println!("Generated C: {} bytes, {} lines", ear_c.len(), ear_c.lines().count());

    // --- Parallel BFS ---
    println!("\n=== Parallel BFS ===\n");
    let bfs = algorithm_library::graph::parallel_bfs();
    println!("Algorithm: {}", bfs.name);
    println!("Memory Model: {}", bfs.memory_model);
    println!("Total Statements: {}", bfs.total_stmts());

    let bfs_c = generator.generate(&bfs);
    println!("Generated C: {} bytes, {} lines", bfs_c.len(), bfs_c.lines().count());

    println!("\n{:-<60}", "");
    println!("All connectivity algorithms compiled successfully.");
}
