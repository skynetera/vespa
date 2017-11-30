// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.node.verification.spec;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.vespa.hosted.node.verification.commons.report.SpecVerificationReport;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfo;
import com.yahoo.vespa.hosted.node.verification.spec.retrievers.HardwareInfo.DiskType;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author sgrostad
 * @author olaaun
 */

public class HardwareNodeComparatorTest {

    private HardwareInfo.Builder actualHardware;
    private HardwareInfo.Builder nodeInfo;

    @Before
    public void setup() {
        actualHardware = new HardwareInfo.Builder()
                .withMinCpuCores(24)
                .withMinMainMemoryAvailableGb(1000)
                .withInterfaceSpeedMbs(10000)
                .withMinDiskAvailableGb(500);

        nodeInfo = new HardwareInfo.Builder()
                .withMinCpuCores(24)
                .withMinMainMemoryAvailableGb(16)
                .withInterfaceSpeedMbs(10000)
                .withMinDiskAvailableGb(500);
    }

    @Test
    public void compare_equal_hardware_should_create_emmpty_json() throws Exception {
        assertSpecVerificationReport("{}");
    }

    @Test
    public void compare_different_amount_of_cores_should_create_json_with_actual_core_amount() throws Exception {
        actualHardware.withMinCpuCores(4);
        nodeInfo.withMinCpuCores(1);
        assertSpecVerificationReport("{\"actualcpuCores\":4}");
    }

    @Test
    public void compare_different_disk_type_should_create_json_with_actual_disk_type() throws Exception {
        actualHardware.withDiskType(DiskType.SLOW);
        nodeInfo.withDiskType(DiskType.FAST);
        assertSpecVerificationReport("{\"actualDiskType\":\"SLOW\"}");
    }

    private void assertSpecVerificationReport(String expectedReportSerialized) throws JsonProcessingException {
        SpecVerificationReport report = HardwareNodeComparator.compare(nodeInfo.build(), actualHardware.build());
        String actualJson = new ObjectMapper().writeValueAsString(report);
        assertEquals(expectedReportSerialized, actualJson);
    }
}
