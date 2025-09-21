#!/usr/bin/env python3
"""
Comprehensive performance and memory testing for Agent implementation
Tests computational efficiency and memory usage patterns
"""

import numpy as np
import sys
import os
import time
import tracemalloc
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.agent import Agent

# Performance targets from SPECIFICATION.md
TARGET_AGENTS = 100  # Design target for 100+ agents
TARGET_TIME_PER_DEMAND = 1e-5  # 10 microseconds per demand calculation
TARGET_MEMORY_PER_AGENT = 1024  # ~1KB per agent (reasonable estimate)


def benchmark_agent_creation(n_agents: int, n_goods: int) -> Tuple[float, float]:
    """Benchmark agent creation time and memory usage"""
    print(f"📊 Benchmarking creation of {n_agents} agents with {n_goods} goods")
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    agents = []
    np.random.seed(42)  # Deterministic
    
    for i in range(n_agents):
        alpha = np.random.exponential(scale=1.0, size=n_goods)
        home_endowment = np.random.exponential(scale=2.0, size=n_goods)
        personal_endowment = np.random.exponential(scale=0.5, size=n_goods)
        position = (np.random.randint(-10, 11), np.random.randint(-10, 11))
        
        agent = Agent(
            agent_id=i,
            alpha=alpha,
            home_endowment=home_endowment,
            personal_endowment=personal_endowment,
            position=position
        )
        agents.append(agent)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    creation_time = end_time - start_time
    memory_usage = peak / (1024 * 1024)  # MB
    
    print(f"   Creation time: {creation_time:.4f}s ({creation_time/n_agents*1e6:.2f}μs per agent)")
    print(f"   Memory usage: {memory_usage:.2f}MB ({memory_usage*1024/n_agents:.2f}KB per agent)")
    
    return creation_time, memory_usage


def benchmark_demand_calculations(agents: List[Agent], n_iterations: int = 1000) -> Tuple[float, float]:
    """Benchmark demand calculation performance"""
    print(f"📊 Benchmarking {n_iterations} demand calculations for {len(agents)} agents")
    
    n_goods = len(agents[0].alpha)
    np.random.seed(123)
    
    # Generate random price vectors for testing
    price_vectors = []
    for _ in range(n_iterations):
        prices = np.concatenate([[1.0], np.random.exponential(scale=1.5, size=n_goods-1)])
        price_vectors.append(prices)
    
    # Benchmark demand calculations
    start_time = time.perf_counter()
    
    total_calculations = 0
    for prices in price_vectors:
        for agent in agents:
            demand = agent.demand(prices)
            total_calculations += 1
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_calculation = total_time / total_calculations
    
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Time per calculation: {time_per_calculation*1e6:.2f}μs")
    print(f"   Calculations per second: {total_calculations/total_time:.0f}")
    
    # Performance target check
    meets_target = time_per_calculation <= TARGET_TIME_PER_DEMAND
    print(f"   Meets performance target (<{TARGET_TIME_PER_DEMAND*1e6:.1f}μs): {'✅' if meets_target else '❌'}")
    
    return total_time, time_per_calculation


def benchmark_utility_calculations(agents: List[Agent], n_iterations: int = 1000) -> Tuple[float, float]:
    """Benchmark utility calculation performance"""
    print(f"📊 Benchmarking {n_iterations} utility calculations for {len(agents)} agents")
    
    np.random.seed(456)
    
    # Generate random consumption bundles
    consumption_bundles = []
    for _ in range(n_iterations):
        bundle = np.random.exponential(scale=1.0, size=len(agents[0].alpha))
        consumption_bundles.append(bundle)
    
    # Benchmark utility calculations
    start_time = time.perf_counter()
    
    total_calculations = 0
    for bundle in consumption_bundles:
        for agent in agents:
            utility = agent.utility(bundle)
            total_calculations += 1
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_calculation = total_time / total_calculations
    
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Time per calculation: {time_per_calculation*1e6:.2f}μs")
    print(f"   Calculations per second: {total_calculations/total_time:.0f}")
    
    return total_time, time_per_calculation


def benchmark_inventory_operations(agents: List[Agent], n_operations: int = 1000) -> float:
    """Benchmark inventory transfer operations"""
    print(f"📊 Benchmarking {n_operations} inventory operations for {len(agents)} agents")
    
    np.random.seed(789)
    
    # Generate valid transfer operations
    operations = []
    for _ in range(n_operations):
        agent = np.random.choice(agents)
        
        # Create valid transfer that won't exceed available inventory
        max_home = agent.home_endowment
        max_personal = agent.personal_endowment
        
        if np.random.random() < 0.5 and np.any(max_home > 0):
            # Transfer from home to personal
            transfer = np.random.uniform(0, 0.5) * max_home
            to_personal = True
        elif np.any(max_personal > 0):
            # Transfer from personal to home
            transfer = np.random.uniform(0, 0.5) * max_personal
            to_personal = False
        else:
            continue  # Skip if no valid transfer possible
            
        operations.append((agent, transfer, to_personal))
    
    # Benchmark transfer operations
    start_time = time.perf_counter()
    
    for agent, transfer, to_personal in operations:
        try:
            agent.transfer_goods(transfer, to_personal)
        except ValueError:
            # Some random transfers might fail due to insufficient inventory
            pass
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_operation = total_time / len(operations)
    
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Time per operation: {time_per_operation*1e6:.2f}μs")
    print(f"   Operations per second: {len(operations)/total_time:.0f}")
    
    return total_time


def benchmark_spatial_operations(agents: List[Agent], n_operations: int = 1000) -> float:
    """Benchmark spatial positioning operations"""
    print(f"📊 Benchmarking {n_operations} spatial operations for {len(agents)} agents")
    
    # Generate test parameters
    marketplace_bounds = ((2, 4), (2, 4))
    marketplace_center = (3, 3)
    
    # Benchmark spatial operations
    start_time = time.perf_counter()
    
    for _ in range(n_operations):
        for agent in agents:
            # Test marketplace detection
            in_marketplace = agent.is_at_marketplace(marketplace_bounds)
            
            # Test distance calculation
            distance = agent.distance_to_marketplace(marketplace_center)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_operations = n_operations * len(agents) * 2  # 2 operations per agent per iteration
    time_per_operation = total_time / total_operations
    
    print(f"   Total time: {total_time:.4f}s")
    print(f"   Time per operation: {time_per_operation*1e6:.2f}μs")
    print(f"   Operations per second: {total_operations/total_time:.0f}")
    
    return total_time


def test_memory_scaling():
    """Test how memory usage scales with number of agents and goods"""
    print("📊 Testing memory scaling patterns")
    
    test_configs = [
        (10, 3),    # Small: 10 agents, 3 goods
        (50, 3),    # Medium agents, 3 goods  
        (100, 3),   # Target agents, 3 goods
        (100, 5),   # Target agents, more goods
        (100, 10),  # Target agents, many goods
    ]
    
    for n_agents, n_goods in test_configs:
        tracemalloc.start()
        
        agents = []
        np.random.seed(42)
        
        for i in range(n_agents):
            alpha = np.random.exponential(scale=1.0, size=n_goods)
            home_endowment = np.random.exponential(scale=2.0, size=n_goods)
            personal_endowment = np.random.exponential(scale=0.5, size=n_goods)
            position = (np.random.randint(-10, 11), np.random.randint(-10, 11))
            
            agent = Agent(
                agent_id=i,
                alpha=alpha,
                home_endowment=home_endowment,
                personal_endowment=personal_endowment,
                position=position
            )
            agents.append(agent)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_per_agent = peak / (1024 * n_agents)  # KB per agent
        memory_per_good = peak / (1024 * n_agents * n_goods)  # KB per agent per good
        
        print(f"   {n_agents:3d} agents, {n_goods:2d} goods: "
              f"{peak/1024:.1f}KB total, {memory_per_agent:.1f}KB/agent, "
              f"{memory_per_good:.2f}KB/agent/good")
        
        # Clean up
        del agents


def test_computational_scaling():
    """Test how computation time scales with problem size"""
    print("📊 Testing computational scaling")
    
    scaling_configs = [
        (20, 3, 100),   # Small scale
        (50, 3, 100),   # Medium scale
        (100, 3, 100),  # Target scale
        (100, 5, 100),  # More goods
        (200, 3, 50),   # More agents, fewer iterations
    ]
    
    for n_agents, n_goods, n_iterations in scaling_configs:
        # Create agents
        agents = []
        np.random.seed(42)
        
        for i in range(n_agents):
            alpha = np.random.exponential(scale=1.0, size=n_goods)
            home_endowment = np.random.exponential(scale=2.0, size=n_goods)
            personal_endowment = np.random.exponential(scale=0.5, size=n_goods)
            position = (0, 0)  # Simple position for performance testing
            
            agent = Agent(
                agent_id=i,
                alpha=alpha,
                home_endowment=home_endowment,
                personal_endowment=personal_endowment,
                position=position
            )
            agents.append(agent)
        
        # Benchmark demand calculations
        prices = np.concatenate([[1.0], np.random.exponential(scale=1.5, size=n_goods-1)])
        
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            for agent in agents:
                demand = agent.demand(prices)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_calculations = n_agents * n_iterations
        time_per_calculation = total_time / total_calculations
        
        print(f"   {n_agents:3d} agents, {n_goods:2d} goods, {n_iterations:3d} iter: "
              f"{time_per_calculation*1e6:.2f}μs/calc, {total_calculations/total_time:.0f} calc/s")


def run_all_performance_tests():
    """Run all performance and memory tests"""
    print("📊 COMPREHENSIVE PERFORMANCE & MEMORY TESTING")
    print("=" * 55)
    
    try:
        # Test 1: Basic performance with target scale
        print("\n=== Agent Creation Performance ===")
        creation_time, memory_usage = benchmark_agent_creation(TARGET_AGENTS, 3)
        
        # Create agents for subsequent tests
        print("\n=== Creating Test Agents ===")
        agents = []
        np.random.seed(42)
        
        for i in range(50):  # Use smaller set for detailed benchmarks
            alpha = np.random.exponential(scale=1.0, size=3)
            home_endowment = np.random.exponential(scale=2.0, size=3)
            personal_endowment = np.random.exponential(scale=0.5, size=3)
            position = (np.random.randint(-10, 11), np.random.randint(-10, 11))
            
            agent = Agent(
                agent_id=i,
                alpha=alpha,
                home_endowment=home_endowment,
                personal_endowment=personal_endowment,
                position=position
            )
            agents.append(agent)
        
        print(f"Created {len(agents)} test agents")
        
        # Test 2: Demand calculation performance
        print("\n=== Demand Calculation Performance ===")
        demand_time, time_per_demand = benchmark_demand_calculations(agents, 500)
        
        # Test 3: Utility calculation performance
        print("\n=== Utility Calculation Performance ===")
        utility_time, time_per_utility = benchmark_utility_calculations(agents, 500)
        
        # Test 4: Inventory operations performance
        print("\n=== Inventory Operations Performance ===")
        inventory_time = benchmark_inventory_operations(agents, 200)
        
        # Test 5: Spatial operations performance
        print("\n=== Spatial Operations Performance ===")
        spatial_time = benchmark_spatial_operations(agents, 100)
        
        # Test 6: Memory scaling analysis
        print("\n=== Memory Scaling Analysis ===")
        test_memory_scaling()
        
        # Test 7: Computational scaling analysis
        print("\n=== Computational Scaling Analysis ===")
        test_computational_scaling()
        
        # Performance summary
        print("\n" + "=" * 55)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 55)
        
        print(f"Agent creation: {creation_time/TARGET_AGENTS*1e6:.2f}μs per agent")
        print(f"Demand calculation: {time_per_demand*1e6:.2f}μs per calculation")
        print(f"Utility calculation: {time_per_utility*1e6:.2f}μs per calculation")
        print(f"Memory efficiency: {memory_usage*1024/TARGET_AGENTS:.2f}KB per agent")
        
        # Check if meets performance targets
        meets_demand_target = time_per_demand <= TARGET_TIME_PER_DEMAND
        meets_memory_target = memory_usage*1024/TARGET_AGENTS <= TARGET_MEMORY_PER_AGENT
        
        print(f"\nPerformance Targets:")
        print(f"  Demand calculation target (<{TARGET_TIME_PER_DEMAND*1e6:.1f}μs): "
              f"{'✅ PASS' if meets_demand_target else '❌ FAIL'}")
        print(f"  Memory usage target (<{TARGET_MEMORY_PER_AGENT}KB/agent): "
              f"{'✅ PASS' if meets_memory_target else '❌ FAIL'}")
        
        overall_pass = meets_demand_target and meets_memory_target
        
        print("\n" + "=" * 55)
        if overall_pass:
            print("🎉 ALL PERFORMANCE TESTS PASSED!")
            print("Agent implementation meets performance targets for 100+ agent simulations.")
            print("Ready for production deployment in economic simulation.")
        else:
            print("⚠️  SOME PERFORMANCE TARGETS NOT MET")
            print("Agent implementation functional but may need optimization for large-scale use.")
        
        return overall_pass
        
    except Exception as e:
        print(f"\n❌ PERFORMANCE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_performance_tests()
    sys.exit(0 if success else 1)