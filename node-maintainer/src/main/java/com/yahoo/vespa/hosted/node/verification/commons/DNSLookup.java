package com.yahoo.vespa.hosted.node.verification.commons;

import sun.net.spi.nameservice.dns.DNSNameService;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.List;

/**
 * Wrapper around DNSNameService that can be easily mocked (Mockito cannot mock DNSNameService because it is final)
 *
 * @author valerijf
 */
public class DNSLookup {

    private final DNSNameService dnsNameService;

    public DNSLookup() {
        try {
            this.dnsNameService = new DNSNameService();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public List<InetAddress> lookupAllHostAddr(String hostname) {
        try {
            return Arrays.asList(dnsNameService.lookupAllHostAddr(hostname));
        } catch (UnknownHostException e) {
            throw new RuntimeException(e);
        }
    }
}
