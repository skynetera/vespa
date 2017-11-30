// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.commons.noderepo;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.Set;

/**
 * Object with the information node repositories has about the node.
 *
 * @author olaaun
 * @author sgrostad
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class NodeRepoJsonModel {

    @JsonProperty("minDiskAvailableGb")
    private double minDiskAvailableGb;
    @JsonProperty("minMainMemoryAvailableGb")
    private double minMainMemoryAvailableGb;
    @JsonProperty("minCpuCores")
    private int minCpuCores;
    @JsonProperty("fastDisk")
    private boolean fastDisk;
    @JsonProperty("ipAddresses")
    private Set<String> ipAddresses;
    @JsonProperty("hostname")
    private String hostname;
    @JsonProperty("hardwareDivergence")
    private String hardwareDivergence;

    public double getMinDiskAvailableGb() {
        return minDiskAvailableGb;
    }

    public double getMinMainMemoryAvailableGb() {
        return minMainMemoryAvailableGb;
    }

    public int getMinCpuCores() {
        return minCpuCores;
    }

    public boolean isFastDisk() {
        return fastDisk;
    }

    public Set<String> getIpAddresses() {
        return Collections.unmodifiableSet(ipAddresses);
    }

    public String getHostname() {
        return hostname;
    }

    public String getHardwareDivergence() {
        return hardwareDivergence;
    }

}
