#!/usr/bin/env python3
"""
Performance test script for RAG system optimizations.
Tests the impact of caching and reduced CAG limit.
"""

import asyncio
import time
import requests
import json
from typing import List, Dict
import statistics

# Test queries - some repeated to test caching
TEST_QUERIES = [
    "ur test query1",
    "ur test query2",
    "ur test query3",  # Repeat for cache test
    "ur test query4",
    "ur test query5",  # Repeat for cache test
    "ur test query6"
]

def test_query_performance(base_url: str = "http://127.0.0.1:8000", num_runs: int = 3):
    """Test performance of multiple queries"""

    print("🚀 Starting RAG Performance Test")
    print("=" * 50)

    all_results = []

    for run in range(num_runs):
        print(f"\n📊 Run {run + 1}/{num_runs}")
        print("-" * 30)

        run_results = []

        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"Query {i}: '{query[:40]}...'")

            start_time = time.time()
            try:
                response = requests.post(
                    f"{base_url}/query",
                    json={"query": query},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    total_time = time.time() - start_time

                    # Extract timing from response (if available)
                    timing_info = {
                        'query': query,
                        'total_time': total_time,
                        'status': 'success',
                        'answerable': result.get('answerable', False),
                        'cag_score': result.get('cag_max_score', 0.0)
                    }

                    run_results.append(timing_info)
                    print(f"  ✅ Success in {total_time:.2f}s (CAG: {result.get('cag_max_score', 0.0):.2f})")
                else:
                    total_time = time.time() - start_time
                    run_results.append({
                        'query': query,
                        'total_time': total_time,
                        'status': 'error',
                        'error_code': response.status_code
                    })
                    print(f"  ❌ Error {response.status_code}: {response.text[:100]}")

            except Exception as e:
                total_time = time.time() - start_time
                run_results.append({
                    'query': query,
                    'total_time': total_time,
                    'status': 'exception',
                    'error': str(e)
                })
                print(f"  ❌ Exception: {str(e)[:100]}")

        all_results.extend(run_results)

    # Analyze results
    analyze_results(all_results, num_runs)

def analyze_results(results: List[Dict], num_runs: int):
    """Analyze and display performance results"""

    print("\n📈 Performance Analysis")
    print("=" * 50)

    # Group by query
    query_groups = {}
    for result in results:
        query = result['query']
        if query not in query_groups:
            query_groups[query] = []
        query_groups[query].append(result)

    # Calculate statistics for each query
    print("\n📊 Per-Query Statistics:")
    print("-" * 80)
    print(f"{'Query':<40} {'Avg Time':>8} {'Min':>8} {'Max':>8} {'Cache'}")
    print("-" * 80)

    for query, query_results in query_groups.items():
        successful_runs = [r for r in query_results if r['status'] == 'success']
        if successful_runs:
            times = [r['total_time'] for r in successful_runs]
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)

            # Check if this query appeared multiple times (cache test)
            is_repeated = len(query_results) > num_runs
            cache_indicator = "🔄" if is_repeated else "🆕"

            print(f"{query[:38]:<40} {avg_time:>8.2f} {min_time:>8.2f} {max_time:>8.2f} {cache_indicator}")

    # Overall statistics
    all_times = [r['total_time'] for r in results if r['status'] == 'success']
    if all_times:
        print("\n🌍 Overall Statistics:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        # CAG performance analysis
        cag_times = []
        for result in results:
            if result['status'] == 'success':
                # Estimate CAG time (rough approximation)
                # In real logs we'd parse this, but for now use empirical data
                cag_times.append(result['total_time'] * 0.4)  # Rough estimate

        if cag_times:
            avg_cag_time = statistics.mean(cag_times)
            print(".2f")
    # Success rate
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    success_rate = (success_count / total_count) * 100

    print(".1f")
    # Cache effectiveness
    repeated_queries = [q for q in TEST_QUERIES if TEST_QUERIES.count(q) > 1]
    if repeated_queries:
        print("\n🔄 Cache Test:")
        print(f"  Repeated queries: {len(set(repeated_queries))}")
        print("  (Check logs for cache hits/misses)")
if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"Testing RAG performance at {base_url}")
    print(f"Number of test runs: {num_runs}")
    print(f"Test queries: {len(TEST_QUERIES)}")

    try:
        test_query_performance(base_url, num_runs)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
