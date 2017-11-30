// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec;

import com.yahoo.log.LogSetup;
import com.yahoo.vespa.hosted.node.verification.commons.CommandExecutor;
import com.yahoo.vespa.hosted.node.verification.commons.HostURLGenerator;
import com.yahoo.vespa.hosted.node.verification.commons.noderepo.NodeRepoInfoRetriever;
import com.yahoo.vespa.hosted.node.verification.commons.noderepo.NodeRepoJsonModel;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfo;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfoRetriever;

import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Creates two HardwareInfo objects, one with spec from node repository and one from spec retrieved at the node.
 * Compares the objects and returns the result.
 *
 * @author olaaun
 * @author sgrostad
 */
public class SpecVerifier {

    private static final Logger logger = Logger.getLogger(SpecVerifier.class.getName());

    private static boolean verifySpec(CommandExecutor commandExecutor, List<URL> nodeInfoUrls, String hostname) throws IOException {
        HardwareInfo expectedHardwareInfo = NodeRepoInfoRetriever.getExpectedHardwareInfo(nodeInfoUrls);

        VerifierSettings verifierSettings = new VerifierSettings(hostname, expectedHardwareInfo.hasIpv6Connectivity());
        HardwareInfoRetriever hardwareInfoRetriever = new HardwareInfoRetriever(commandExecutor, verifierSettings);
        HardwareInfo actualHardware = hardwareInfoRetriever.retrieve();

        SpecVerificationReport specVerificationReport = makeVerificationReport(actualHardware, nodeRepoJsonModel);
        Reporter.reportSpecVerificationResults(specVerificationReport, nodeInfoUrls);
        return specVerificationReport.isValidSpec();
    }

    protected static SpecVerificationReport makeVerificationReport(HardwareInfo actualHardware, NodeRepoJsonModel nodeRepoJsonModel) {
        SpecVerificationReport specVerificationReport = HardwareNodeComparator.compare(, actualHardware);
        IPAddressRetriever ipAddressRetriever = new IPAddressRetriever(ipAddresses);
        ipAddressRetriever.reportFaultyIpAddresses(nodeRepoJsonModel, specVerificationReport);
        return specVerificationReport;
    }

    public static void main(String[] args) {
        LogSetup.initVespaLogging("spec-verifier");
        if (args.length == 0) {
            throw new IllegalStateException("Expected config server URL as parameter");
        }
        try {
            CommandExecutor commandExecutor = new CommandExecutor();
            String hostname = getHostname(commandExecutor);
            List<URL> nodeInfoUrls = HostURLGenerator.generateNodeInfoUrl(args[0], hostname);
            verifySpec(commandExecutor, nodeInfoUrls, hostname);
        } catch (IOException e) {
            logger.log(Level.WARNING, e.getMessage());
        }
    }

    private static String getHostname(CommandExecutor commandExecutor) throws IOException {
        List<String> output = commandExecutor.executeCommand("hostname");
        if (output.size() == 1) {
            return output.get(0);
        }
        throw new IOException("Unexpected output from \"hostname\" command.");
    }
}
