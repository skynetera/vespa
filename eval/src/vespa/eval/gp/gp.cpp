// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "gp.h"
#include <algorithm>
#include <vespa/vespalib/util/stringfmt.h>

namespace vespalib::gp {

namespace {

Value get(const Input &input, const std::vector<Value> &values, Program::Ref ref) {
    return ref.is_input() ? input[ref.in_idx()] : values[ref.op_idx()];
}

size_t get(const std::vector<size_t> &sizes, Program::Ref ref) {
    return ref.is_input() ? 1 : sizes[ref.op_idx()];
}

} // namespace vespalib::gp::<unnamed>

void
Program::assert_valid(Ref ref, size_t limit) const
{
    assert(ref.is_input() != ref.is_operation());
    if (ref.is_input()) {
        assert(ref.in_idx() < _in_cnt);
    }
    if (ref.is_operation()) {
        assert(ref.op_idx() < limit);
    }
}

Program::Ref
Program::add_op(size_t code, Ref lhs, Ref rhs)
{
    size_t op_idx = _program.size();
    assert(code <= _repo.max_op());
    assert_valid(lhs, op_idx);
    assert_valid(rhs, op_idx);
    _program.emplace_back(code, lhs, rhs);
    return Ref::op(op_idx);
}

Program::Ref
Program::add_forward(Ref ref)
{
    return add_op(0, ref, Ref::nop());
}

void
Program::grow(Random &rnd, size_t op_cnt)
{
    assert((op_cnt / _out_cnt) >= 1);
    assert((op_cnt % _out_cnt) == 0);
    while (_program.size() < op_cnt) {
        size_t op_idx = _program.size();    
        add_op(rnd_op(rnd),
               rnd_ref(rnd, op_idx),
               rnd_ref(rnd, op_idx));
    }
}

void
Program::mutate(Random &rnd)
{
    size_t mut_idx = rnd.get(0, _program.size() - 1);
    Op &op = _program[mut_idx];
    size_t sel = rnd.get(0,2);
    if (sel == 0) {
        op.code = rnd_op(rnd);
    } else if (sel == 1) {
        op.lhs = rnd_ref(rnd, mut_idx);
    } else {
        assert(sel == 2);
        op.rhs = rnd_ref(rnd, mut_idx);
    }
}

size_t
Program::get_cost(size_t slot) const
{
    size_t offset = (slot * _out_cnt);
    assert((offset + _out_cnt) <= _program.size());
    size_t cost = 0;
    std::vector<bool> done(_program.size(), false);
    std::vector<Ref> todo;
    for (size_t i = 0; i < _out_cnt; ++i) {
        todo.push_back(Ref::op(offset + i));
    }
    while (!todo.empty()) {
        Ref ref = todo.back();
        todo.pop_back();
        if (ref.is_operation()) {
            if (!done[ref.op_idx()]) {
                const Op &op = _program[ref.op_idx()];
                cost += _repo.cost_of(op.code);
                todo.push_back(op.lhs);
                if (op.code > 0) {
                    todo.push_back(op.rhs);
                }
                done[ref.op_idx()] = true;
            }
        }
    }
    return cost;
}

size_t
Program::size_of(Ref ref) const
{
    assert_valid(ref, _program.size());
    if (ref.is_input()) {
        return 1;
    }
    std::vector<size_t> sizes;
    for (size_t i = 0; i <= ref.op_idx(); ++i) {
        const Op &op = _program[i];
        if (op.code == 0) {
            sizes.push_back(get(sizes, op.lhs)); // forward
        } else {
            sizes.push_back(1 + get(sizes, op.lhs) + get(sizes, op.rhs));
        }
    }
    return sizes.back();
}

vespalib::string
Program::as_string(Ref ref) const
{
    assert_valid(ref, _program.size());
    size_t expr_size = size_of(ref);
    if (expr_size > 9000) {
        // its over 9000!
        return vespalib::make_string("expr(%zu nodes)", expr_size);
    } else if (ref.is_input()) {
        return vespalib::make_string("i%zu", ref.in_idx());
    } else {
        const Op &my_op = _program[ref.op_idx()];   
        if (my_op.code == 0) {
            return as_string(my_op.lhs); // forward
        } else {
            return vespalib::make_string("%s(%s,%s)", _repo.name_of(my_op.code).c_str(),
                                         as_string(my_op.lhs).c_str(), as_string(my_op.rhs).c_str());
        }
    }
}

Result
Program::execute(const Input &input) const
{
    Result result;
    std::vector<Value> out;
    std::vector<Value> values;
    for (const Op &op: _program) {
        Value value = _repo.perform(op.code,
                                    get(input, values, op.lhs),
                                    get(input, values, op.rhs));
        values.push_back(value);
        out.push_back(value);
        if (out.size() == _out_cnt) {
            if (values.size() == _program.size()) {
                // HACK: only return last one
                result.push_back(out);
            }
            out.clear();
        }
    }
    return result;
}

void
Program::handle_feedback(const Feedback &feedback)
{
    assert(feedback.size() == num_alternatives());
    if (num_alternatives() == 1) {
        _best_slot = (_program.size() - _out_cnt) / _out_cnt;
        _stats = Stats(feedback[0], get_cost(_best_slot), _stats.born);
    } else {
        _stats = Stats(feedback[0], get_cost(0), _stats.born);
        _best_slot = 0;
        for (size_t i = 1; i < feedback.size(); ++i) {
            Stats stats(feedback[i], get_cost(i), _stats.born);
            if (stats < _stats) {
                _stats = stats;
                _best_slot = i;
            }
        }
    }
}

void
Population::grow()
{
    while (_programs.size() < _params.pop_cnt) {
        _programs.emplace_back(_repo, _params.in_cnt, _params.out_cnt, _gen);
        _programs.back().grow(_rnd, _params.op_cnt);
        _repo.find_weakness(_programs.back());
    }
    std::sort(_programs.begin(), _programs.end());
}

void
Population::print_stats() const
{
    const Program::Stats &best = _programs.front().stats();
    const Program::Stats &worst = _programs.back().stats();
    fprintf(stderr, "[%zu] best(weakness=%g,cost=%zu,age=%zu), "
            "worst(weakness=%g,cost=%zu,age=%zu)\n", _gen,
            best.weakness, best.cost, (_gen - best.born),
            worst.weakness, worst.cost, (_gen - worst.born));
}

const Program &
Population::select(size_t limit)
{
    return _programs[std::min(_rnd.get(0, limit - 1),
                              _rnd.get(0, limit - 1))];
}

Program
Population::mutate(const Program &a)
{
    Program new_prog = a;
    do {
        new_prog.mutate(_rnd);
    } while(_rnd.get(0,99) < 66);
    new_prog.reborn(_gen);
    return new_prog;
}

void
Population::tick()
{
    ++_gen;
    size_t apex_cnt = (_params.pop_cnt / 10);
    while (_programs.size() > apex_cnt) {
        _programs.pop_back();
    }
    while (_programs.size() < _params.pop_cnt) {
        _programs.push_back(mutate(select(apex_cnt)));
        _repo.find_weakness(_programs.back());
    }
    std::sort(_programs.begin(), _programs.end());
}

} // namespace vespalib::gp
