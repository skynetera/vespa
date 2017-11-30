// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

/**
 * @author olaaun
 * @author sgrostad
 */
public interface HardwareRetriever {

    /**
     * Should retrieve spec from some part of the hardware, and store the result in
     * the HardwareInfo.Builder instance passed to the method
     */
    void updateInfo(HardwareInfo.Builder hardwareInfoBuilder);

}
