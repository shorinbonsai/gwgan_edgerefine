use graph_refiner::engine::GeneticOptimizer; 
use std::fs;

// Helper to generate a dummy optimizer
fn setup_optimizer() -> GeneticOptimizer {
    let pop_size = 20;
    let gene_len = 10;
    // [CHANGE] Use GeneticOptimizer::new instead of GraphRefiner
    let mut optimizer = GeneticOptimizer::new(pop_size);
    
    // Setup a simple ring graph: 0-1-2-3-0
    let edges = vec![(0,1), (1,2), (2,3), (3,0)];
    optimizer.initialize_population(4, edges, 42, gene_len);
    
    optimizer.set_op_weights(vec![1.0; 9]);
    
    let empty_dist = vec![vec![0.0; 10]];
    let empty_stats = vec![0.0; 10];
    // Create a vector of ones for standard deviation to avoid division by zero (NaN)
    let unity_std = vec![1.0; 10];
    
    optimizer.set_targets(
        empty_dist.clone(), empty_stats.clone(), unity_std.clone(), // degree: std=1.0
        empty_dist.clone(), empty_stats.clone(), unity_std.clone(), // clustering: std=1.0
        empty_dist.clone(), empty_stats.clone(), unity_std.clone(), // spectral: std=1.0
        (1.0, 1.0, 1.0), 
        (1.0, 1.0, 1.0),
        100,
    );
    
    optimizer
}

#[test]
fn test_initialization() {
    let optimizer = setup_optimizer();
    let best_edges = optimizer.get_best_edges();
    assert!(!best_edges.is_empty(), "Graph should be initialized with edges");
}

#[test]
fn test_evolution_runs() {
    let mut optimizer = setup_optimizer();
    let fitness = optimizer.evolve(5, 42); // Run 5 generations
    assert!(fitness.is_finite(), "Fitness should be a finite number");
    assert!(fitness >= 0.0, "Fitness should be non-negative");
}

#[test]
fn test_logging_and_saving() {
    let mut optimizer = setup_optimizer();
    optimizer.evolve(5, 123);

    let log_file = "test_log.csv";
    let res_file = "test_results.txt";

    // Test Saving Logs
    let res_log = optimizer.save_logs(log_file);
    assert!(res_log.is_ok(), "Should save logs successfully");
    
    // Verify Log Content
    let log_content = fs::read_to_string(log_file).expect("Log file should exist");
    assert!(log_content.contains("Generation,MinFitness"), "Log header missing");
    
    // Test Saving Results
    let res_save = optimizer.save_results(res_file);
    assert!(res_save.is_ok(), "Should save results successfully");
    
    // Verify Result Content
    let res_content = fs::read_to_string(res_file).expect("Result file should exist");
    assert!(res_content.contains("Best Individual Results"), "Result header missing");

    // Cleanup
    let _ = fs::remove_file(log_file);
    let _ = fs::remove_file(res_file);
}




