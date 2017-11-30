// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import java.net.InetAddress;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * All information the different retrievers retrieve is stored as a HardwareInfo object.
 *
 * @author olaaun
 * @author sgrostad
 */
public class HardwareInfo {

    private final double minDiskAvailableGb;
    private final double minMainMemoryAvailableGb;
    private final int minCpuCores;
    private final double interfaceSpeedMbs;
    private final DiskType diskType;
    private final Set<InetAddress> ipAddresses;
    private final boolean hasIpv4Interface;
    private final boolean hasIpv6Interface;
    private final boolean hasIpv6Connectivity;

    // For testing
    public HardwareInfo() {
        this(0, 0, 0, 0, DiskType.UNKNOWN, Collections.emptySet(), false, false, false);
    }

    public HardwareInfo(double minDiskAvailableGb, double minMainMemoryAvailableGb, int minCpuCores,
                        double interfaceSpeedMbs, DiskType diskType, Set<InetAddress> ipAddresses,
                        boolean hasIpv4Interface, boolean hasIpv6Interface, boolean hasIpv6Connectivity) {
        Objects.requireNonNull(diskType);
        Objects.requireNonNull(ipAddresses);

        this.minDiskAvailableGb = minDiskAvailableGb;
        this.minMainMemoryAvailableGb = minMainMemoryAvailableGb;
        this.minCpuCores = minCpuCores;
        this.interfaceSpeedMbs = interfaceSpeedMbs;
        this.ipAddresses = Collections.unmodifiableSet(new HashSet<>(ipAddresses));
        this.diskType = diskType;
        this.hasIpv4Interface = hasIpv4Interface;
        this.hasIpv6Interface = hasIpv6Interface;
        this.hasIpv6Connectivity = hasIpv6Connectivity;
    }

    public double getMinDiskAvailableGb() {
        return minDiskAvailableGb;
    }

    public double getMinMainMemoryAvailableGb() {
        return minMainMemoryAvailableGb;
    }

    public int getMinCpuCores() {
        return minCpuCores;
    }

    public double getInterfaceSpeedMbs() {
        return interfaceSpeedMbs;
    }

    public DiskType getDiskType() {
        return diskType;
    }

    public Set<InetAddress> getIpAddresses() {
        return ipAddresses;
    }

    public boolean hasIpv4Interface() {
        return hasIpv4Interface;
    }

    public boolean hasIpv6Interface() {
        return hasIpv6Interface;
    }

    public boolean hasIpv6Connectivity() {
        return hasIpv6Connectivity;
    }


    public enum DiskType {SLOW, FAST, UNKNOWN}

    public static class Builder {
        private double minDiskAvailableGb;
        private double minMainMemoryAvailableGb;
        private int minCpuCores;
        private double interfaceSpeedMbs;
        private DiskType diskType = DiskType.UNKNOWN;
        private Set<InetAddress> ipAddresses;
        private boolean hasIpv4Interface;
        private boolean hasIpv6Interface;
        private boolean hasIpv6Connectivity;

        public Builder withMinDiskAvailableGb(double minDiskAvailableGb) {
            this.minDiskAvailableGb = minDiskAvailableGb;
            return this;
        }

        public Builder withMinMainMemoryAvailableGb(double minMainMemoryAvailableGb) {
            this.minMainMemoryAvailableGb = minMainMemoryAvailableGb;
            return this;
        }

        public Builder withMinCpuCores(int minCpuCores) {
            this.minCpuCores = minCpuCores;
            return this;
        }

        public Builder withInterfaceSpeedMbs(double interfaceSpeedMbs) {
            this.interfaceSpeedMbs = interfaceSpeedMbs;
            return this;
        }

        public Builder withDiskType(DiskType diskType) {
            this.diskType = diskType;
            return this;
        }

        public Builder withIpAddresses(Set<InetAddress> ipAddresses) {
            this.ipAddresses = ipAddresses;
            return this;
        }

        public Builder withIpv4Interface(boolean hasIpv4Interface) {
            this.hasIpv4Interface = hasIpv4Interface;
            return this;
        }

        public Builder withIpv6Interface(boolean hasIpv6Interface) {
            this.hasIpv6Interface = hasIpv6Interface;
            return this;
        }

        public Builder withIpv6Connectivity(boolean hasIpv6Connectivity) {
            this.hasIpv6Connectivity = hasIpv6Connectivity;
            return this;
        }

        public HardwareInfo build() {
            return new HardwareInfo(minDiskAvailableGb, minMainMemoryAvailableGb, minCpuCores, interfaceSpeedMbs,
                    diskType, ipAddresses, hasIpv4Interface, hasIpv6Interface, hasIpv6Connectivity);
        }
    }
}
