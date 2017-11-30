// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import com.google.common.net.InetAddresses;
import com.yahoo.vespa.hosted.node.verification.commons.DNSLookup;
import com.yahoo.vespa.hosted.node.verification.commons.parser.ParseResult;
import com.yahoo.vespa.hosted.node.verification.mock.MockCommandExecutor;
import com.yahoo.vespa.hosted.node.verification.spec.VerifierSettings;
import org.junit.Test;

import java.io.IOException;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * @author sgrostad
 * @author olaaun
 */
public class NetRetrieverTest {

    private static final String RESOURCE_PATH = "src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/";
    private static final String NET_FIND_INTERFACE = RESOURCE_PATH + "ifconfig";
    private static final String NET_CHECK_INTERFACE_SPEED = RESOURCE_PATH + "eth0";
    private static final String VALID_PING_RESPONSE = RESOURCE_PATH + "validpingresponse";
    private static final String INVALID_PING_RESPONSE = RESOURCE_PATH + "invalidpingresponse";
    private static final String PING_SEARCH_WORD = "\\d+\\.?\\d*";
    private static final double DELTA = 0.1;

    private final String hostname = "test123.some.domain.tld";
    private final VerifierSettings verifierSettings = new VerifierSettings(hostname, true);
    private final DNSLookup dnsLookup = mock(DNSLookup.class);
    private final MockCommandExecutor commandExecutor = new MockCommandExecutor();
    private final NetRetriever net = new NetRetriever(commandExecutor, dnsLookup, verifierSettings);

    @Test
    public void updateInfo_should_store_ipv4_ipv6_interface_and_interface_speed() {
        HardwareInfo.Builder hardwareInfoBuilder = mock(HardwareInfo.Builder.class);
        commandExecutor.addCommand("cat " + NET_FIND_INTERFACE);
        commandExecutor.addCommand("cat " + NET_CHECK_INTERFACE_SPEED);
        commandExecutor.addCommand("cat " + VALID_PING_RESPONSE);
        List<InetAddress> ipAddresses = Stream.of("::1", "::2", "127.0.0.1")
                .map(InetAddresses::forString)
                .collect(Collectors.toList());

        when(dnsLookup.lookupAllHostAddr(eq(hostname))).thenReturn(ipAddresses);
        net.updateInfo(hardwareInfoBuilder);

        verify(hardwareInfoBuilder, times(1)).withIpv4Interface(true);
        verify(hardwareInfoBuilder, times(1)).withIpv6Interface(true);
        verify(hardwareInfoBuilder, times(1)).withIpv6Connectivity(true);
        verify(hardwareInfoBuilder, times(1)).withInterfaceSpeedMbs(1000);
        verify(hardwareInfoBuilder, times(1)).withIpAddresses(eq(new HashSet<>(ipAddresses)));
        verifyNoMoreInteractions(hardwareInfoBuilder);
    }

    @Test
    public void findInterface_valid_input() {
        commandExecutor.addCommand("cat " + NET_FIND_INTERFACE);
        List<ParseResult> actualParseResults = net.findInterface();

        List<ParseResult> expectedParseResults = new ArrayList<>();
        expectedParseResults.add(new ParseResult("inet", "inet"));
        expectedParseResults.add(new ParseResult("inet6", "inet6"));
        assertEquals(expectedParseResults, actualParseResults);
    }

    @Test
    public void findInterfaceSpeed_valid_input() {
        commandExecutor.addCommand("cat " + NET_CHECK_INTERFACE_SPEED);
        List<ParseResult> parseResults = net.findInterfaceSpeed();

        ParseResult expectedParseResults = new ParseResult("Speed", "1000Mb/s");
        assertEquals(Collections.singletonList(expectedParseResults), parseResults);
    }

    @Test
    public void parseNetInterface_get_ipv_from_ifconfigNotIpv6_testFile() throws IOException {
        List<String> mockOutput = MockCommandExecutor.readFromFile(NET_FIND_INTERFACE + "NoIpv6");
        List<ParseResult> parseResults = net.parseNetInterface(mockOutput);
        List<ParseResult> expextedParseResults = Collections.singletonList(new ParseResult("inet", "inet"));
        assertEquals(expextedParseResults, parseResults);
    }

    @Test
    public void parseInterfaceSpeed_get_interfaceSpeed_from_eth0_testFile() throws IOException {
        List<String> mockOutput = MockCommandExecutor.readFromFile("src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/eth0");
        ParseResult parseResult = net.parseInterfaceSpeed(mockOutput);
        ParseResult expectedParseResult = new ParseResult("Speed", "1000Mb/s");
        assertEquals(expectedParseResult, parseResult);
    }

    @Test
    public void stripInterfaceSpeed_should_return_correct_double() {
        String interfaceSpeedToConvert = "1000Mb/s";
        double expectedInterfaceSpeed = 1000;
        double actualInterfaceSpeed = NetRetriever.getInterfaceSpeed(interfaceSpeedToConvert);
        assertEquals(expectedInterfaceSpeed, actualInterfaceSpeed, DELTA);
    }

    @Test
    public void parsePingResponse_valid_ping_response_should_return_ipv6_connectivity() throws IOException {
        List<String> mockCommandOutput = MockCommandExecutor.readFromFile(VALID_PING_RESPONSE);
        ParseResult parseResult = net.parsePingResponse(mockCommandOutput);
        String expectedPing = "0";
        assertEquals(expectedPing, parseResult.getValue());
    }

    @Test
    public void parsePingResponse_invalid_ping_response_should_throw_IOException() throws IOException {
        List<String> mockCommandOutput = MockCommandExecutor.readFromFile(INVALID_PING_RESPONSE);
        try {
            net.parsePingResponse(mockCommandOutput);
            fail("Expected an IOException to be thrown");
        } catch (IllegalArgumentException e) {
            String expectedExceptionMessage = "Failed to parse ping output.";
            assertEquals(expectedExceptionMessage, e.getMessage());
        }
    }

    @Test
    public void getIpv6Connectivity_valid_ping_response_should_return_ipv6_connectivity() {
        ParseResult parseResult = new ParseResult(PING_SEARCH_WORD, "0");
        assertTrue(net.getIpv6Connectivity(parseResult));
    }

    @Test
    public void getIpv6Connectivity_invalid_ping_response_should_return_no_ipv6_connectivity_1() {
        ParseResult parseResult = new ParseResult(PING_SEARCH_WORD, "100");
        assertFalse(net.getIpv6Connectivity(parseResult));
    }

    @Test
    public void getIpv6Connectivity_invalid_ping_response_should_return_no_ipv6_connectivity_2() {
        ParseResult parseResult = new ParseResult(PING_SEARCH_WORD, "invalid");
        assertFalse(net.getIpv6Connectivity(parseResult));
    }
}
