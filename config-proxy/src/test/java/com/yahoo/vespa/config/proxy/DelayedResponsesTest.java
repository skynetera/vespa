// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.config.proxy;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

/**
 * @author hmusum
 * @since 5.1.9
 */
public class DelayedResponsesTest {

    @Test
    public void basic() throws InterruptedException {
        ConfigTester tester = new ConfigTester();
        DelayedResponses responses = new DelayedResponses(new ConfigProxyStatistics());
        DelayedResponse delayedResponse = new DelayedResponse(tester.createRequest("foo", "id", "bar", 10));
        responses.add(delayedResponse);

        assertThat(responses.size(), is(1));
        assertThat(responses.responses().take(), is(delayedResponse));
        assertThat(responses.size(), is(0));

        responses.add(delayedResponse);
        assertThat(responses.size(), is(1));
        responses.remove(delayedResponse);
        assertThat(responses.size(), is(0));
    }

}
