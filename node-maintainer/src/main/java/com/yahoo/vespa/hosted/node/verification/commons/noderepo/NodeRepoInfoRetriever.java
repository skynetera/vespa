// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.commons.noderepo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.net.InetAddresses;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfo;

import java.io.IOException;
import java.net.Inet4Address;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.URL;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Parse JSON from node repository and stores information as a NodeRepoJsonModel object.
 *
 * @author olaaun
 * @author sgrostad
 */
public class NodeRepoInfoRetriever {

    private static final Logger logger = Logger.getLogger(NodeRepoInfoRetriever.class.getName());

    private static NodeRepoJsonModel retrieve(List<URL> nodeInfoUrls) throws IOException {
        NodeRepoJsonModel nodeRepoJsonModel;
        ObjectMapper objectMapper = new ObjectMapper();
        for (URL nodeInfoURL : nodeInfoUrls) {
            try {
                nodeRepoJsonModel = objectMapper.readValue(nodeInfoURL, NodeRepoJsonModel.class);
                return nodeRepoJsonModel;
            } catch (IOException e) {
                logger.log(Level.WARNING, "Failed to parse JSON from config server: " + nodeInfoURL.toString(), e);
            }
        }
        throw new IOException("Failed to parse JSON from all possible config servers.");
    }

    public static HardwareInfo getExpectedHardwareInfo(List<URL> nodeInfoUrls) throws IOException {
        NodeRepoJsonModel nodeRepoJsonModel = retrieve(nodeInfoUrls);

        Set<InetAddress> ipAddresses = nodeRepoJsonModel.getIpAddresses().stream()
                .map(InetAddresses::forString)
                .collect(Collectors.toSet());

        boolean hasIpv4 = ipAddresses.stream().anyMatch(ip -> ip instanceof Inet4Address);
        boolean hasIpv6 = ipAddresses.stream().anyMatch(ip -> ip instanceof Inet6Address);

        return new HardwareInfo(nodeRepoJsonModel.getMinDiskAvailableGb(),
                nodeRepoJsonModel.getMinMainMemoryAvailableGb(),
                nodeRepoJsonModel.getMinCpuCores(),
                1000,
                nodeRepoJsonModel.isFastDisk() ? HardwareInfo.DiskType.FAST : HardwareInfo.DiskType.SLOW,
                ipAddresses,
                hasIpv4,
                hasIpv6,
                hasIpv6);
    }
}

