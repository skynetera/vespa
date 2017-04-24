// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Copyright (C) 2003 Fast Search & Transfer ASA
// Copyright (C) 2003 Overture Services Norway AS

#pragma once


namespace search {

/*
 * Simple random generator based on lrand48() spec.
 */
class Rand48
{
private:
    uint64_t _state;
public:
    void
    srand48(long seed)
    {
        _state = ((static_cast<uint64_t>(seed & 0xffffffffu)) << 16) + 0x330e;
    }

    Rand48(void)
        : _state(0)
    {
        srand48(0x1234abcd);
    };
    void iterate(void) {
        _state = (UINT64_C(0x5DEECE66D) * _state + 0xb) &
                 UINT64_C(0xFFFFFFFFFFFF);
    }
    /*
     * Return value from 0 to 2^31 - 1
     */
    long lrand48(void) {
        iterate();
        return static_cast<long>(_state >> 17);
    }
};

} // namespace search

