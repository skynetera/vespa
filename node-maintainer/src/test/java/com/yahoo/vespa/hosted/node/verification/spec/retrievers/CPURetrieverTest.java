// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import com.yahoo.vespa.hosted.node.verification.commons.parser.ParseResult;
import com.yahoo.vespa.hosted.node.verification.mock.MockCommandExecutor;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * @author sgrostad
 * @author olaaun
 */

public class CPURetrieverTest {

    private static final String FILENAME = "src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/cpuinfoTest";

    private final MockCommandExecutor commandExecutor = new MockCommandExecutor();
    private final CPURetriever cpu = new CPURetriever(commandExecutor);

    @Test
    public void updateInfo_should_write_numOfCpuCores_to_hardware_info() throws Exception {
        HardwareInfo.Builder hardwareInfoBuilder = mock(HardwareInfo.Builder.class);
        commandExecutor.addCommand("cat " + FILENAME);
        cpu.updateInfo(hardwareInfoBuilder);

        verify(hardwareInfoBuilder, times(1)).withMinCpuCores(4);
        verifyNoMoreInteractions(hardwareInfoBuilder);
    }

    @Test
    public void parseCPUInfoFile_should_return_valid_List() throws IOException {
        List<String> commandOutput = MockCommandExecutor.readFromFile(FILENAME);
        List<ParseResult> ParseResults = cpu.parseCPUInfoFile(commandOutput);
        String expectedSearchWord = "cpu MHz";
        String expectedValue = "2493.821";

        assertEquals(expectedSearchWord, ParseResults.get(0).getSearchWord());
        assertEquals(expectedValue, ParseResults.get(0).getValue());

        assertEquals(expectedSearchWord, ParseResults.get(1).getSearchWord());
        assertEquals(expectedValue, ParseResults.get(1).getValue());

        assertEquals(expectedSearchWord, ParseResults.get(2).getSearchWord());
        assertEquals(expectedValue, ParseResults.get(2).getValue());

        assertEquals(expectedSearchWord, ParseResults.get(3).getSearchWord());
        assertEquals(expectedValue, ParseResults.get(3).getValue());
    }

    @Test
    public void setCpuCores_counts_cores_correctly() {
        List<ParseResult> parseResults = new ArrayList<>();
        parseResults.add(new ParseResult("cpu MHz", "2000"));
        parseResults.add(new ParseResult("cpu MHz", "2000"));
        parseResults.add(new ParseResult("cpu MHz", "2000"));

        assertEquals(3, cpu.countCpuCores(parseResults));
    }

}
