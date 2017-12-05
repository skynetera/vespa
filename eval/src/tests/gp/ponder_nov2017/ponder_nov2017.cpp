// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/vespalib/util/signalhandler.h>
#include <vespa/eval/gp/gp.h>
#include <limits.h>
#include <algorithm>

using namespace vespalib;
using namespace vespalib::gp;

// Inspired by the great and sometimes frustrating puzzles posed to us
// by IBM; what about automatically evolving a solution instead of
// figuring it out on our own. Turns out GP is no free lunch, but
// rather a strange and interesting adventure all of its own...

// problem: https://www.research.ibm.com/haifa/ponderthis/challenges/November2017.html
// solution: https://www.research.ibm.com/haifa/ponderthis/solutions/November2017.html

// illegal div/mod will result in 0
bool div_ok(int a, int b) {
    if ((a == INT_MIN) && (b == -1)) {
        return false;
    }
    return (b != 0);
}
int my_add(int a, int b) { return a + b; }
int my_sub(int a, int b) { return a - b; }
int my_mul(int a, int b) { return a * b; }
int my_div(int a, int b) { return div_ok(a, b) ? (a / b) : 0; }
int my_mod(int a, int b) { return div_ok(a, b) ? (a % b) : 0; }
int my_pow(int a, int b) { return pow(a,b); }
int my_and(int a, int b) { return a & b; }
int my_or(int a, int b)  { return a | b; }
int my_xor(int a, int b) { return a ^ b; }


// 2*2*6 = 24 (number of slots)
// 6*6*6/24 = 9 (target events per slot)
struct Dist {
    std::vector<int> slots;
    Dist() : slots(24, 0) {}
    void sample(int x, int y, int z) {
        int post_x = (x & 1);
        int post_y = (y & 1);
        int post_z = (size_t(z) % 6);
        ASSERT_GREATER_EQUAL(post_x, 0);
        ASSERT_GREATER_EQUAL(post_y, 0);
        ASSERT_GREATER_EQUAL(post_z, 0);
        ASSERT_LESS(post_x, 2);
        ASSERT_LESS(post_y, 2);
        ASSERT_LESS(post_z, 6);
        int slot = (post_z<<2) | (post_y<<1) | (post_x);
        ASSERT_LESS(size_t(slot), slots.size());
        ++slots[slot];
    }
    size_t error() const {
        size_t err = 0;
        for (int cnt: slots) {
            err += (std::max(cnt, 9) - std::min(cnt, 9));
        }
        return err;
    }
};

Feedback find_weakness(const MultiFunction &fun) {
    std::vector<Dist> state(fun.num_alternatives());
    for (int d1 = 1; d1 <= 6; ++d1) {
        for (int d2 = 1; d2 <= 6; ++d2) {
            for (int d3 = 1; d3 <= 6; ++d3) {
                Input input({d1, d2, d3});
                std::sort(input.begin(), input.end());
                if (fun.num_inputs() == 6) {
                    // add const values for hand-crafted case
                    input.push_back(2);
                    input.push_back(1502);
                    input.push_back(70677);
                }
                Result result = fun.execute(input);
                ASSERT_EQUAL(result.size(), state.size());
                for (size_t i = 0; i < result.size(); ++i) {
                    const Output &output = result[i];
                    ASSERT_EQUAL(output.size(), 3u);
                    state[i].sample(output[0], output[1], output[2]);
                }
            }
        }
    }
    Feedback feedback;
    for (const Dist &dist: state) {
        feedback.push_back(dist.error());
    }
    return feedback;
}

OpRepo my_repo() {
    return OpRepo(find_weakness)
        .add("add", my_add)  // 1
        .add("sub", my_sub)  // 2
        .add("mul", my_mul)  // 3
        .add("div", my_div)  // 4
        .add("mod", my_mod)  // 5
        .add("pow", my_pow)  // 6
        .add("and", my_and)  // 7
        .add("or",  my_or)   // 8
        .add("xor", my_xor); // 9
}

// Featured solution (Bert Dobbelaere):
//
// d=2**(((c-a)*(c+a))/2)
//     x=(1502/d)%2
//     y=(70677/d)%2
//     z=(a+b+c)%6+1

const size_t add_id = 1;
const size_t sub_id = 2;
const size_t mul_id = 3;
const size_t div_id = 4;
const size_t pow_id = 6;

using Ref = Program::Ref;
using Op = Program::Op;

TEST("evaluating hand-crafted solution") {
    // constants are modeled as inputs
    Program prog(my_repo(), 6, 3, 0);
    auto a = Ref::in(0);                   // a
    auto b = Ref::in(1);                   // b
    auto c = Ref::in(2);                   // c
    auto k1 = Ref::in(3);                  // 2
    auto k2 = Ref::in(4);                  // 1502
    auto k3 = Ref::in(5);                  // 70677
    // --- slot 0
    auto _1 = prog.add_op(sub_id, c, a);   // _1 = c-a
    auto _2 = prog.add_op(add_id, c, a);   // _2 = c+a
    auto _3 = prog.add_op(mul_id, _1, _2); // _3 = (c-a)*(c+a)
    // --- slot 1 (zero-cost forward layer)
    _1 = prog.add_forward(_1);
    _2 = prog.add_forward(_2);
    _3 = prog.add_forward(_3);
    // --- slot 2
    auto _4 = prog.add_op(div_id, _3, k1); // _4 = ((c-a)*(c+a))/2
    auto d = prog.add_op(pow_id, k1, _4);  // d = 2**(((c-a)*(c+a))/2)
    auto _5 = prog.add_op(add_id, a, b);   // _5 = a+b
    // --- slot 3
    auto x = prog.add_op(div_id, k2, d);   // x = 1502/d
    auto y = prog.add_op(div_id, k3, d);   // y = 70677/d
    auto z = prog.add_op(add_id, _5, c);   // z = a+b+c
    // '%2' (for x and y) and '%6+1' (for z) done outside program
    //--- verify sub-expressions
    EXPECT_EQUAL(prog.as_string(a), "i0");
    EXPECT_EQUAL(prog.as_string(k2), "i4");
    EXPECT_EQUAL(prog.as_string(d), "pow(i3,div(mul(sub(i2,i0),add(i2,i0)),i3))");
    EXPECT_EQUAL(prog.as_string(x), "div(i4,pow(i3,div(mul(sub(i2,i0),add(i2,i0)),i3)))");
    EXPECT_EQUAL(prog.as_string(y), "div(i5,pow(i3,div(mul(sub(i2,i0),add(i2,i0)),i3)))");
    EXPECT_EQUAL(prog.as_string(z), "add(add(i0,i1),i2)");
    //--- verify (expression) sizes
    EXPECT_EQUAL(prog.size_of(a), 1u);
    EXPECT_EQUAL(prog.size_of(k2), 1u);
    EXPECT_EQUAL(prog.size_of(d), 11u);
    EXPECT_EQUAL(prog.size_of(x), 13u);
    EXPECT_EQUAL(prog.size_of(y), 13u);
    EXPECT_EQUAL(prog.size_of(z), 5u);
    //--- verify costs
    EXPECT_EQUAL(prog.get_cost(0), 3u);
    EXPECT_EQUAL(prog.get_cost(1), 3u);
    EXPECT_EQUAL(prog.get_cost(2), 6u);
    EXPECT_EQUAL(prog.get_cost(3), 9u);
    //--- evaluate
    prog.handle_feedback(find_weakness(prog));
    EXPECT_EQUAL(prog.stats().weakness, 0.0);
    EXPECT_EQUAL(prog.stats().cost, 9u);
    EXPECT_EQUAL(prog._best_slot, 3u);
}

void maybe_newline(bool &partial_line) {
    if (partial_line) {
        fprintf(stderr, "\n");
        partial_line = false;
    }
}

Program try_evolve(const Params &params, size_t max_ticks) {
    Population population(params, my_repo(), Random().make_seed());
    bool partial_line = false;
    size_t ticks = 0;
    for (; ticks < max_ticks; ++ticks) {
        if (SignalHandler::INT.check()) {
            maybe_newline(partial_line);
            fprintf(stderr, "<INT>\n");
            break;
        } else if ((ticks % 100) == 0) {
            maybe_newline(partial_line);
            population.print_stats();
        } else if ((ticks % 2) == 0) {
            fprintf(stderr, ".");
            partial_line = true;
        }
        population.tick();
    }
    maybe_newline(partial_line);
    Program::Stats best = population._programs[0].stats();
    fprintf(stderr, "best stats after %zu ticks: (weakness=%g,cost=%zu)\n",
            ticks, best.weakness, best.cost);
    return population._programs[0];
}

Params my_params() {
    size_t in_cnt = 3;
    size_t out_cnt = 3;
    size_t op_cnt = 33;
    size_t pop_cnt = 100; // 10 + 90
    return Params(in_cnt, out_cnt, op_cnt, pop_cnt);
}

const size_t num_ticks = 10000000;

TEST("trying to evolve a solution automatically") {
    Program best = try_evolve(my_params(), num_ticks);
    size_t offset = (best._best_slot * 3);
    auto x = Ref::op(offset);
    auto y = Ref::op(offset + 1);
    auto z = Ref::op(offset + 2);
    fprintf(stderr, "x(size=%zu): %s\n", best.size_of(x), best.as_string(x).c_str());
    fprintf(stderr, "y(size=%zu): %s\n", best.size_of(y), best.as_string(y).c_str());
    fprintf(stderr, "z(size=%zu): %s\n", best.size_of(z), best.as_string(z).c_str());
}

TEST_MAIN() {
    SignalHandler::INT.hook();
    TEST_RUN_ALL();
}
