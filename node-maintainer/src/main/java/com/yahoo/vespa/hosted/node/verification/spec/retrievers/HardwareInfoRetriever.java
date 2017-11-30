// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import com.yahoo.vespa.hosted.node.verification.commons.CommandExecutor;
import com.yahoo.vespa.hosted.node.verification.commons.DNSLookup;
import com.yahoo.vespa.hosted.node.verification.spec.VerifierSettings;

import java.util.ArrayList;
import java.util.List;

/**
 * Makes a HardwareInfo object and calls all the retrievers for this object.
 *
 * @author olaaun
 * @author sgrostad
 */
public class HardwareInfoRetriever {

    private final List<HardwareRetriever> infoList = new ArrayList<>();

    public HardwareInfoRetriever(CommandExecutor commandExecutor, VerifierSettings verifierSettings) {
        infoList.add(new CPURetriever(commandExecutor));
        infoList.add(new MemoryRetriever(commandExecutor));
        infoList.add(new DiskRetriever(commandExecutor));
        infoList.add(new NetRetriever(commandExecutor, new DNSLookup(), verifierSettings));
    }

    public HardwareInfo retrieve() {
        HardwareInfo.Builder hardwareInfoBuilder = new HardwareInfo.Builder();

        for (HardwareRetriever hardwareInfoType : infoList) {
            hardwareInfoType.updateInfo(hardwareInfoBuilder);
        }

        return hardwareInfoBuilder.build();
    }
}
