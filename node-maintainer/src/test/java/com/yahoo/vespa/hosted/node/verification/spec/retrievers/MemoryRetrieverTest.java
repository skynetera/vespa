// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import com.yahoo.vespa.hosted.node.verification.commons.parser.ParseResult;
import com.yahoo.vespa.hosted.node.verification.mock.MockCommandExecutor;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.AdditionalMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * @author sgrostad
 * @author olaaun
 */
public class MemoryRetrieverTest {

    private static final String FILENAME = "src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/meminfoTest";
    private final double DELTA = 0.1;

    private MockCommandExecutor commandExecutor = new MockCommandExecutor();
    private MemoryRetriever memory = new MemoryRetriever(commandExecutor);

    @Test
    public void updateInfo_should_set_memory_available_in_hardwareInfo() {
        HardwareInfo.Builder hardwareInfoBuilder = mock(HardwareInfo.Builder.class);
        commandExecutor.addCommand("cat " + FILENAME);
        memory.updateInfo(hardwareInfoBuilder);

        verify(hardwareInfoBuilder, times(1)).withMinMainMemoryAvailableGb(eq(4.042128, DELTA));
        verifyNoMoreInteractions(hardwareInfoBuilder);
    }

    @Test
    public void parseMemInfoFile_should_return_valid_parseResult() throws IOException {
        List<String> commandOutput = MockCommandExecutor.readFromFile(FILENAME);
        ParseResult parseResult = memory.parseMemInfoFile(commandOutput);
        ParseResult expectedParseResult = new ParseResult("MemTotal", "4042128 kB");
        assertEquals(expectedParseResult, parseResult);
    }

    @Test
    public void convertToGB_valid_input() {
        String testTotMem = "4042128";
        double expectedTotMem = 4.042128;
        assertEquals(expectedTotMem, MemoryRetriever.convertKBToGB(testTotMem), DELTA);
    }

}
