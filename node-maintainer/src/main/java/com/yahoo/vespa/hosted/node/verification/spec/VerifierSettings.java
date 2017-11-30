// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec;

/**
 * Contains information on what spec should be verified or not.
 * 
 * @author sgrostad
 * @author olaaun
 */
public class VerifierSettings {

    private final String hostname;
    private final boolean checkIPv6;

    public VerifierSettings(String hostname, boolean checkIPv6) {
        this.checkIPv6 = checkIPv6;
        this.hostname = hostname;
    }

    public String getHostname() {
        return hostname;
    }

    public boolean isCheckIPv6() {
        return checkIPv6;
    }

}
