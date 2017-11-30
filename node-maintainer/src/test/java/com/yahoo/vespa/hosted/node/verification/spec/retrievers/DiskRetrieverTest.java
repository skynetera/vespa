// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec.retrievers;

import com.yahoo.vespa.hosted.node.verification.commons.parser.ParseResult;
import com.yahoo.vespa.hosted.node.verification.mock.MockCommandExecutor;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfo.DiskType;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.mockito.AdditionalMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * @author sgrostad
 * @author olaaun
 */

public class DiskRetrieverTest {

    private static final String CAT_RESOURCE_PATH = "cat src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/";
    private static final double DELTA = 0.1;
    private static final double expectedSize = 1759.84;

    private final MockCommandExecutor commandExecutor = new MockCommandExecutor();
    private final DiskRetriever diskRetriever = new DiskRetriever(commandExecutor);

    @Test
    public void updateInfo_should_store_diskType_and_diskSize_in_hardware_info() {
        HardwareInfo.Builder hardwareInfoBuilder = mock(HardwareInfo.Builder.class);
        commandExecutor.addCommand(CAT_RESOURCE_PATH + "DiskTypeFastDisk");
        commandExecutor.addCommand(CAT_RESOURCE_PATH + "filesize");
        diskRetriever.updateInfo(hardwareInfoBuilder);

        verify(hardwareInfoBuilder, times(1)).withDiskType(DiskType.FAST);
        verify(hardwareInfoBuilder, times(1)).withMinDiskAvailableGb(eq(expectedSize, DELTA));
        verifyNoMoreInteractions(hardwareInfoBuilder);
    }

    @Test
    public void parseDiskType_should_find_fast_disk() throws Exception {
        List<String> mockOutput = commandExecutor.outputFromString("Name  Rota \nsda 0");
        assertEquals(DiskType.FAST, diskRetriever.parseDiskType(mockOutput));
    }

    @Test
    public void parseDiskType_should_not_find_fast_disk() throws Exception {
        List<String> mockOutput = commandExecutor.outputFromString("Name  Rota \nsda 1");
        assertEquals(DiskType.SLOW, diskRetriever.parseDiskType(mockOutput));
    }

    @Test
    public void parseDiskType_with_invalid_outputstream_does_not_contain_searchword_should_throw_exception() throws Exception {
        List<String> mockOutput = commandExecutor.outputFromString("Name  Rota");
        try {
            diskRetriever.parseDiskType(mockOutput);
            fail("Should have thrown IOException when outputstream doesn't contain search word");
        } catch (Exception e) {
            String expectedExceptionMessage = "Parsing for disk type failed";
            assertEquals(expectedExceptionMessage, e.getMessage());
        }

    }

    @Test
    public void parseDiskSize_should_find_size_from_file_and_insert_into_parseResult() throws Exception {
        String filepath = "src/test/java/com/yahoo/vespa/hosted/node/verification/spec/resources/filesize";
        List<String> mockOutput = MockCommandExecutor.readFromFile(filepath);
        List<ParseResult> parseResults = diskRetriever.parseDiskSize(mockOutput);
        ParseResult expectedParseResult1 = new ParseResult("Size", "799.65");
        assertEquals(expectedParseResult1, parseResults.get(0));
        ParseResult expectedParseResult2 = new ParseResult("Size", "960.19");
        assertEquals(expectedParseResult2, parseResults.get(1));
    }

    @Test
    public void setDiskType_invalid_ParseResult_should_set_fastDisk_to_invalid() {
        ParseResult parseResult = new ParseResult("Invalid", "Invalid");
        assertEquals(DiskType.UNKNOWN, diskRetriever.getDiskType(parseResult));
    }

}
