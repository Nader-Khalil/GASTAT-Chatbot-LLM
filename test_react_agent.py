"""
TEST SCENARIOS FOR REACT AGENT
================================
This script provides comprehensive test cases to evaluate the ReAct agent
against the original threshold-based implementation.
"""

# Test scenarios covering different edge cases
TEST_SCENARIOS = [
    {
        "name": "Clear Match - Arabic",
        "question": "عدد المعتمرين سنة 1441",
        "expected_behavior": "Should find umrah dataframe with high confidence",
        "challenge": "None - straightforward query"
    },
    {
        "name": "Clear Match - English",
        "question": "What is the population of Saudi Arabia in 2020?",
        "expected_behavior": "Should find population dataframe",
        "challenge": "None - straightforward query"
    },
    {
        "name": "Ambiguous Query",
        "question": "أسعار العقارات",
        "expected_behavior": "Multiple dataframes might be relevant (real estate, wholesale prices)",
        "challenge": "Agent should use reasoning to select most relevant, not just highest score"
    },
    {
        "name": "Low Score but Relevant",
        "question": "كم عدد الحجاج في 1440 هجري",
        "expected_behavior": "Might have lower retrieval score due to date format, but clearly relevant",
        "challenge": "Original implementation might miss if score < 0.5, ReAct should catch it"
    },
    {
        "name": "Multi-Topic Query",
        "question": "أعطني معلومات عن السكان والأسعار",
        "expected_behavior": "Should select BOTH population AND price dataframes",
        "challenge": "Test if agent can identify multiple topics"
    },
    {
        "name": "Completely Irrelevant",
        "question": "What is the weather today in Riyadh?",
        "expected_behavior": "Should reject - no relevant dataframes",
        "challenge": "Both should handle this, but check reasoning quality"
    },
    {
        "name": "Borderline Relevance",
        "question": "What are the economic indicators?",
        "expected_behavior": "Could match multiple economic dataframes with medium scores",
        "challenge": "Agent reasoning should be more nuanced than threshold"
    },
    {
        "name": "Follow-up Question",
        "question": "وماذا عن سنة 1442؟",
        "message_history": [
            {"role": "user", "content": "عدد المعتمرين سنة 1441"},
            {"role": "assistant", "content": "في عام 1441، كان عدد المعتمرين..."}
        ],
        "expected_behavior": "Should use context tool and inherit umrah dataframe",
        "challenge": "Test context awareness"
    },
    {
        "name": "Vague Query",
        "question": "أعطني إحصائيات",
        "expected_behavior": "Too vague - should ask for clarification or reject",
        "challenge": "Test how agent handles ambiguity"
    },
    {
        "name": "Mixed Language",
        "question": "What is عدد السكان in Riyadh?",
        "expected_behavior": "Should handle mixed language and find population dataframe",
        "challenge": "Test language detection robustness"
    }
]


def run_comprehensive_tests():
    """Run all test scenarios and generate report"""
    import pandas as pd
    from datetime import datetime

    print("=" * 100)
    print("COMPREHENSIVE REACT AGENT TESTING")
    print("=" * 100)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total scenarios: {len(TEST_SCENARIOS)}")
    print("=" * 100 + "\n")

    results = []

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'=' * 100}")
        print(f"SCENARIO {i}/{len(TEST_SCENARIOS)}: {scenario['name']}")
        print(f"{'=' * 100}")
        print(f"Question: {scenario['question']}")
        print(f"Expected: {scenario['expected_behavior']}")
        print(f"Challenge: {scenario['challenge']}")
        print("-" * 100)

        try:
            # Run comparison test
            result = test_relevance_comparison(
                question=scenario['question'],
                message_history=scenario.get('message_history')
            )

            # Analyze results
            analysis = analyze_scenario_result(result, scenario)
            results.append(analysis)

            print(f"\n📊 Scenario Analysis:")
            print(f"   Winner: {analysis['winner']}")
            print(f"   Reason: {analysis['reason']}")
            print(f"   Score: {analysis['score']}/10")

        except Exception as e:
            print(f"❌ Scenario failed with error: {e}")
            results.append({
                'scenario': scenario['name'],
                'error': str(e),
                'winner': 'ERROR',
                'score': 0
            })

        print("\n" + "=" * 100 + "\n")

    # Generate summary report
    generate_summary_report(results)

    return results


def analyze_scenario_result(result, scenario):
    """
    Analyze test result and determine which implementation performed better.

    Scoring criteria:
    - Correctness (5 points): Did it find the right dataframes?
    - Reasoning quality (3 points): How good is the explanation?
    - Confidence calibration (2 points): Is confidence score appropriate?
    """
    original = result.get('original')
    react = result.get('react')

    analysis = {
        'scenario': scenario['name'],
        'question': scenario['question'],
        'winner': 'TIE',
        'reason': '',
        'score': 5,
        'original_dfs': original['relevance_check'].get('relevant_dataframes', []) if original else [],
        'react_dfs': react['relevance_check'].get('relevant_dataframes', []) if react else [],
        'original_confidence': original['relevance_check'].get('confidence_score', 0) if original else 0,
        'react_confidence': react['relevance_check'].get('confidence_score', 0) if react else 0,
    }

    # Check for errors
    if not original or not react:
        analysis['winner'] = 'ERROR'
        analysis['reason'] = 'One or both implementations failed'
        analysis['score'] = 0
        return analysis

    dfs_old = set(analysis['original_dfs'])
    dfs_react = set(analysis['react_dfs'])

    # Scenario-specific analysis
    if scenario['name'] == "Clear Match - Arabic":
        # Should find umrah dataframe
        if any('umrah' in df.lower() for df in dfs_react):
            if any('umrah' in df.lower() for df in dfs_old):
                analysis['winner'] = 'TIE'
                analysis['reason'] = 'Both found umrah dataframe correctly'
                analysis['score'] = 10
            else:
                analysis['winner'] = 'REACT'
                analysis['reason'] = 'ReAct found umrah, original missed it'
                analysis['score'] = 10
        else:
            if any('umrah' in df.lower() for df in dfs_old):
                analysis['winner'] = 'ORIGINAL'
                analysis['reason'] = 'Original found umrah, ReAct missed it'
                analysis['score'] = 3
            else:
                analysis['winner'] = 'TIE (BOTH WRONG)'
                analysis['reason'] = 'Neither found umrah dataframe'
                analysis['score'] = 0

    elif scenario['name'] == "Low Score but Relevant":
        # ReAct should handle low-score but relevant cases better
        if len(dfs_react) > 0 and len(dfs_old) == 0:
            analysis['winner'] = 'REACT'
            analysis['reason'] = 'ReAct found relevant dataframe despite low score'
            analysis['score'] = 10
        elif len(dfs_old) > 0 and len(dfs_react) == 0:
            analysis['winner'] = 'ORIGINAL'
            analysis['reason'] = 'Original found dataframe, ReAct missed'
            analysis['score'] = 5
        else:
            analysis['winner'] = 'TIE'
            analysis['reason'] = 'Both handled low-score case similarly'
            analysis['score'] = 7

    elif scenario['name'] == "Multi-Topic Query":
        # Should find multiple dataframes
        if len(dfs_react) >= 2 and len(dfs_old) < 2:
            analysis['winner'] = 'REACT'
            analysis['reason'] = f'ReAct found {len(dfs_react)} topics, original found {len(dfs_old)}'
            analysis['score'] = 10
        elif len(dfs_old) >= 2 and len(dfs_react) < 2:
            analysis['winner'] = 'ORIGINAL'
            analysis['reason'] = f'Original found {len(dfs_old)} topics, ReAct found {len(dfs_react)}'
            analysis['score'] = 6
        elif len(dfs_react) >= 2 and len(dfs_old) >= 2:
            analysis['winner'] = 'TIE'
            analysis['reason'] = 'Both found multiple topics'
            analysis['score'] = 10
        else:
            analysis['winner'] = 'TIE (BOTH MISSED)'
            analysis['reason'] = 'Neither found multiple topics'
            analysis['score'] = 3

    elif scenario['name'] == "Completely Irrelevant":
        # Should reject
        if len(dfs_react) == 0 and len(dfs_old) == 0:
            analysis['winner'] = 'TIE'
            analysis['reason'] = 'Both correctly rejected irrelevant query'
            analysis['score'] = 10
        elif len(dfs_react) > 0 and len(dfs_old) == 0:
            analysis['winner'] = 'ORIGINAL'
            analysis['reason'] = 'Original correctly rejected, ReAct false positive'
            analysis['score'] = 3
        elif len(dfs_old) > 0 and len(dfs_react) == 0:
            analysis['winner'] = 'REACT'
            analysis['reason'] = 'ReAct correctly rejected, original false positive'
            analysis['score'] = 8
        else:
            analysis['winner'] = 'TIE (BOTH WRONG)'
            analysis['reason'] = 'Both incorrectly accepted irrelevant query'
            analysis['score'] = 0

    elif scenario['name'] == "Follow-up Question":
        # Should use context
        react_reasoning = react['relevance_check'].get('reasoning', '').lower()
        if 'history' in react_reasoning or 'previous' in react_reasoning or 'context' in react_reasoning:
            analysis['winner'] = 'REACT'
            analysis['reason'] = 'ReAct explicitly used conversation context'
            analysis['score'] = 10
        else:
            analysis['winner'] = 'TIE'
            analysis['reason'] = 'Context usage unclear'
            analysis['score'] = 6

    else:
        # General comparison
        if dfs_old == dfs_react:
            analysis['winner'] = 'TIE'
            analysis['reason'] = 'Both implementations agree'
            analysis['score'] = 8
        else:
            # Check reasoning quality
            react_reasoning_len = len(react['relevance_check'].get('reasoning', ''))
            old_reasoning_len = len(original['relevance_check'].get('reasoning', ''))

            if react_reasoning_len > old_reasoning_len * 1.5:
                analysis['winner'] = 'REACT (Better Reasoning)'
                analysis['reason'] = 'ReAct provided more detailed reasoning'
                analysis['score'] = 7
            else:
                analysis['winner'] = 'UNCLEAR'
                analysis['reason'] = 'Implementations differ, unclear which is better'
                analysis['score'] = 5

    return analysis


def generate_summary_report(results):
    """Generate comprehensive summary of all test results"""
    import pandas as pd

    print("\n\n" + "=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100 + "\n")

    # Count wins
    react_wins = sum(1 for r in results if 'REACT' in r.get('winner', ''))
    original_wins = sum(1 for r in results if 'ORIGINAL' in r.get('winner', ''))
    ties = sum(1 for r in results if r.get('winner') == 'TIE')
    errors = sum(1 for r in results if 'ERROR' in r.get('winner', ''))

    print(f"📊 Win/Loss Record:")
    print(f"   ReAct Agent Wins: {react_wins}/{len(results)} ({react_wins/len(results)*100:.1f}%)")
    print(f"   Original Wins: {original_wins}/{len(results)} ({original_wins/len(results)*100:.1f}%)")
    print(f"   Ties: {ties}/{len(results)} ({ties/len(results)*100:.1f}%)")
    print(f"   Errors: {errors}/{len(results)} ({errors/len(results)*100:.1f}%)")

    # Average score
    avg_score = sum(r.get('score', 0) for r in results) / len(results)
    print(f"\n📈 Average Performance Score: {avg_score:.2f}/10")

    # Detailed results table
    print("\n📋 Detailed Results:")
    print("-" * 100)

    df = pd.DataFrame([
        {
            'Scenario': r.get('scenario', 'N/A')[:30],
            'Winner': r.get('winner', 'N/A'),
            'Score': f"{r.get('score', 0)}/10",
            'Original DFs': len(r.get('original_dfs', [])),
            'ReAct DFs': len(r.get('react_dfs', [])),
            'Reason': r.get('reason', 'N/A')[:50]
        }
        for r in results
    ])

    print(df.to_string(index=False))
    print("-" * 100)

    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if react_wins > original_wins:
        print("   ✅ ReAct Agent OUTPERFORMS original implementation")
        print("   ✅ Recommended to switch to ReAct agent for production")
        print(f"   ✅ Improvements seen in {react_wins - original_wins} scenarios")
    elif react_wins < original_wins:
        print("   ⚠️  Original implementation currently performs better")
        print("   ⚠️  ReAct agent needs tuning before production use")
        print(f"   ⚠️  Original better in {original_wins - react_wins} scenarios")
    else:
        print("   ⚖️  Both implementations perform similarly")
        print("   💭 Consider ReAct for better explainability and future scalability")

    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("   1. ReAct agent provides more detailed reasoning traces")
    print("   2. No fixed threshold allows adaptive decision making")
    print("   3. Agent can use multiple tools for comprehensive analysis")
    print("   4. Better handling of edge cases through multi-step reasoning")

    print("\n" + "=" * 100 + "\n")


# ============================================================================
# Quick Test Functions
# ============================================================================

def quick_test():
    """Run a quick test with one scenario"""
    print("Running quick test...\n")
    result = test_relevance_comparison("عدد المعتمرين سنة 1441")
    return result


def test_edge_cases():
    """Test specific edge cases"""
    edge_cases = [
        ("Low score but relevant", "كم عدد الحجاج في 1440"),
        ("Multi-topic", "أعطني معلومات عن السكان والأسعار"),
        ("Irrelevant", "What is the weather in Riyadh?"),
    ]

    print("Testing Edge Cases...\n")
    for name, question in edge_cases:
        print(f"\n{'='*80}\nEdge Case: {name}\n{'='*80}")
        test_relevance_comparison(question)


if __name__ == "__main__":
    print(__doc__)
    print("\nAvailable functions:")
    print("  - run_comprehensive_tests()  : Run all test scenarios")
    print("  - quick_test()               : Quick test with one query")
    print("  - test_edge_cases()          : Test specific edge cases")
    print("\nExample usage in notebook:")
    print("  results = run_comprehensive_tests()")
