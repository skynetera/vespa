// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.commons;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Makes the URL used to retrieve the JSON from the node repository with information about the node's spec.
 *
 * @author olaaun
 * @author sgrostad
 */
public class HostURLGenerator {

    private static final String NODE_HOSTNAME_PREFIX = "/nodes/v2/node/";
    private static final String PORT_NUMBER = ":4080";
    private static final String HTTP = "http://";
    private static final String PARSE_ALL_HOSTNAMES_REGEX = ",";
    private static final String PROTOCOL_REGEX = "^(https?|file)://.*$";

    public static List<URL> generateNodeInfoUrl(String commaSeparatedUrls, String hostname) throws IOException {
        List<URL> nodeInfoUrls = new ArrayList<>();
        String[] configServerHostNames = commaSeparatedUrls.split(PARSE_ALL_HOSTNAMES_REGEX);
        for (String configServerHostName : configServerHostNames) {
            nodeInfoUrls.add(buildNodeInfoURL(configServerHostName, hostname));
        }
        return nodeInfoUrls;
    }


    static URL buildNodeInfoURL(String configServerHostName, String nodeHostName) throws MalformedURLException {
        if (configServerHostName.matches(PROTOCOL_REGEX)) {
            return new URL(configServerHostName + NODE_HOSTNAME_PREFIX + nodeHostName);
        }
        return new URL(HTTP + configServerHostName + PORT_NUMBER + NODE_HOSTNAME_PREFIX + nodeHostName);
    }

}
