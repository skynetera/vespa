// Copyright 2016 Yahoo Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.orchestrator;

import com.google.inject.Inject;
import com.yahoo.config.provision.ApplicationId;
import com.yahoo.log.LogLevel;
import com.yahoo.vespa.applicationmodel.ApplicationInstance;
import com.yahoo.vespa.applicationmodel.ApplicationInstanceReference;
import com.yahoo.vespa.applicationmodel.ClusterId;
import com.yahoo.vespa.applicationmodel.HostName;
import com.yahoo.vespa.applicationmodel.ServiceCluster;
import com.yahoo.vespa.orchestrator.controller.ClusterControllerClient;
import com.yahoo.vespa.orchestrator.controller.ClusterControllerClientFactory;
import com.yahoo.vespa.orchestrator.controller.ClusterControllerState;
import com.yahoo.vespa.orchestrator.controller.ClusterControllerStateResponse;
import com.yahoo.vespa.orchestrator.model.VespaModelUtil;
import com.yahoo.vespa.orchestrator.policy.BatchHostStateChangeDeniedException;
import com.yahoo.vespa.orchestrator.policy.HostStateChangeDeniedException;
import com.yahoo.vespa.orchestrator.policy.HostedVespaPolicy;
import com.yahoo.vespa.orchestrator.policy.Policy;
import com.yahoo.vespa.orchestrator.status.ApplicationInstanceStatus;
import com.yahoo.vespa.orchestrator.status.HostStatus;
import com.yahoo.vespa.orchestrator.status.MutableStatusRegistry;
import com.yahoo.vespa.orchestrator.status.StatusService;
import com.yahoo.vespa.service.monitor.ServiceMonitorStatus;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * @author oyving
 * @author smorgrav
 */
public class OrchestratorImpl implements Orchestrator {


    private static final Logger log = Logger.getLogger(OrchestratorImpl.class.getName());

    private final Policy policy;
    private final StatusService statusService;
    private final InstanceLookupService instanceLookupService;
    private final int serviceMonitorConvergenceLatencySeconds;
    private final ClusterControllerClientFactory clusterControllerClientFactory;

    @Inject
    public OrchestratorImpl(ClusterControllerClientFactory clusterControllerClientFactory,
                            StatusService statusService,
                            OrchestratorConfig orchestratorConfig,
                            InstanceLookupService instanceLookupService)
    {
        this(new HostedVespaPolicy(clusterControllerClientFactory),
                clusterControllerClientFactory,
                statusService,
                instanceLookupService,
                orchestratorConfig.serviceMonitorConvergenceLatencySeconds());
    }

    public OrchestratorImpl(Policy policy,
                            ClusterControllerClientFactory clusterControllerClientFactory,
                            StatusService statusService,
                            InstanceLookupService instanceLookupService,
                            int serviceMonitorConvergenceLatencySeconds)
    {
        this.policy = policy;
        this.clusterControllerClientFactory = clusterControllerClientFactory;
        this.statusService = statusService;
        this.serviceMonitorConvergenceLatencySeconds = serviceMonitorConvergenceLatencySeconds;
        this.instanceLookupService = instanceLookupService;

    }

    @Override
    public HostStatus getNodeStatus(HostName hostName) throws HostNameNotFoundException {
        return getNodeStatus(getApplicationInstance(hostName).reference(),hostName);
    }

    @Override
    public void resume(HostName hostName) throws HostStateChangeDeniedException, HostNameNotFoundException {
       /*
        * When making a state transition to this state, we have to consider that if the host has been in
        * ALLOWED_TO_BE_DOWN state, services on the host may recently have been stopped (and, presumably, started).
        * Service monitoring may not have had enough time to detect that services were stopped,
        * and may therefore mistakenly report services as up, even if they still haven't initialized and
        * are not yet ready for serving. Erroneously reporting both host and services as up causes a race
        * where services on other hosts may be stopped prematurely. A delay here ensures that service
        * monitoring will have had time to catch up. Since we don't want do the delay with the lock held,
        * and the host status service's locking functionality does not support something like condition
        * variables or Object.wait(), we break out here, releasing the lock before delaying.
        */
        sleep(serviceMonitorConvergenceLatencySeconds, TimeUnit.SECONDS);

        ApplicationInstance<ServiceMonitorStatus> appInstance = getApplicationInstance(hostName);

        try (MutableStatusRegistry statusRegistry = statusService.lockApplicationInstance_forCurrentThreadOnly(appInstance.reference())) {
            final HostStatus currentHostState = statusRegistry.getHostStatus(hostName);

            if (HostStatus.NO_REMARKS == currentHostState) {
                return;
            }

            ApplicationInstanceStatus appStatus = statusService.forApplicationInstance(appInstance.reference()).getApplicationInstanceStatus();
            if (appStatus == ApplicationInstanceStatus.NO_REMARKS) {
                policy.releaseSuspensionGrant(appInstance, hostName, statusRegistry);
            }
        }
    }

    @Override
    public void suspend(HostName hostName) throws HostStateChangeDeniedException, HostNameNotFoundException {
        ApplicationInstance<ServiceMonitorStatus> appInstance = getApplicationInstance(hostName);


        try (MutableStatusRegistry hostStatusRegistry = statusService.lockApplicationInstance_forCurrentThreadOnly(appInstance.reference())) {
            final HostStatus currentHostState = hostStatusRegistry.getHostStatus(hostName);

            if (HostStatus.ALLOWED_TO_BE_DOWN == currentHostState) {
                return;
            }

            ApplicationInstanceStatus appStatus = statusService.forApplicationInstance(appInstance.reference()).getApplicationInstanceStatus();
            if (appStatus == ApplicationInstanceStatus.NO_REMARKS) {
                policy.grantSuspensionRequest(appInstance, hostName, hostStatusRegistry);
            }
        }
    }

    @Override
    public ApplicationInstanceStatus getApplicationInstanceStatus(ApplicationId appId) throws ApplicationIdNotFoundException {
        ApplicationInstanceReference appRef = OrchestratorUtil.toApplicationInstanceReference(appId,instanceLookupService);
        return statusService.forApplicationInstance(appRef).getApplicationInstanceStatus();
    }

    @Override
    public Set<ApplicationId> getAllSuspendedApplications() {
        Set<ApplicationInstanceReference> refSet = statusService.getAllSuspendedApplications();
        return refSet.stream().map(OrchestratorUtil::toApplicationId).collect(Collectors.toSet());
    }

    @Override
    public void resume(ApplicationId appId) throws ApplicationIdNotFoundException, ApplicationStateChangeDeniedException {
        setApplicationStatus(appId, ApplicationInstanceStatus.NO_REMARKS);
    }

    @Override
    public void suspend(ApplicationId appId) throws ApplicationIdNotFoundException, ApplicationStateChangeDeniedException {
        setApplicationStatus(appId, ApplicationInstanceStatus.ALLOWED_TO_BE_DOWN);
    }

    @Override
    public void suspendAll(HostName parentHostname, List<HostName> hostNames)
            throws BatchHostStateChangeDeniedException, BatchHostNameNotFoundException, BatchInternalErrorException {
        try {
            hostNames = sortHostNamesForSuspend(hostNames);
        } catch (HostNameNotFoundException e) {
            throw new BatchHostNameNotFoundException(parentHostname, hostNames, e);
        }

        try {
            for (HostName hostName : hostNames) {
                try {
                    suspend(hostName);
                } catch (HostStateChangeDeniedException e) {
                    throw new BatchHostStateChangeDeniedException(parentHostname, hostNames, e);
                } catch (HostNameNotFoundException e) {
                    // Should never get here since since we would have received HostNameNotFoundException earlier.
                    throw new BatchHostNameNotFoundException(parentHostname, hostNames, e);
                } catch (RuntimeException e) {
                    throw new BatchInternalErrorException(parentHostname, hostNames, e);
                }
            }
        } catch (Exception e) {
            rollbackSuspendAll(hostNames, e);
            throw e;
        }
    }

    private void rollbackSuspendAll(List<HostName> orderedHostNames, Exception exception) {
        List<HostName> reverseOrderedHostNames = new ArrayList<>(orderedHostNames);
        Collections.reverse(reverseOrderedHostNames);
        for (HostName hostName : reverseOrderedHostNames) {
            try {
                resume(hostName);
            } catch (HostStateChangeDeniedException | HostNameNotFoundException | RuntimeException e) {
                // We're forced to ignore these since we're already rolling back a suspension.
                exception.addSuppressed(e);
            }
        }
    }

    /**
     * PROBLEM
     * Take the example of 2 Docker hosts:
     *  - Docker host 1 has two nodes A1 and B1, belonging to the application with
     *    a globally unique ID A and B, respectively.
     *  - Similarly, Docker host 2 has two nodes running content nodes A2 and B2,
     *    and we assume both A1 and A2 (and B1 and B2) have services within the same service cluster.
     *
     * Suppose both Docker hosts wanting to reboot, and
     *  - Docker host 1 asks to suspend A1 and B1, while
     *  - Docker host 2 asks to suspend B2 and A2.
     *
     * The Orchestrator may allow suspend of A1 and B2, before requesting the suspension of B1 and A2.
     * None of these can be suspended (assuming max 1 suspended content node per content cluster),
     * and so both requests for suspension will fail.
     *
     * Note that it's not a deadlock - both client will fail immediately and resume both A1 and B2 before
     * responding to the client, and if host 1 asks later w/o host 2 asking at the same time,
     * it will be given permission to suspend. However if both hosts were to request in lock-step,
     * there would be starvation. And in general, it would fail requests for suspension more
     * than necessary.
     *
     * SOLUTION
     * The solution we're using is to order the hostnames by the globally unique application instance ID,
     * e.g. hosted-vespa:routing:dev:ci-corp-us-east-1:default. In the example above, it would guarantee
     * Docker host 2 would ensure ask to suspend B2 before A2. We take care of that ordering here.
     */
    List<HostName> sortHostNamesForSuspend(List<HostName> hostNames) throws HostNameNotFoundException {
        Map<HostName, ApplicationInstanceReference> applicationReferences = new HashMap<>(hostNames.size());
        for (HostName hostName : hostNames) {
            ApplicationInstance<?> appInstance = getApplicationInstance(hostName);
            applicationReferences.put(hostName, appInstance.reference());
        }

        return hostNames.stream()
                .sorted((leftHostname, rightHostname) -> compareHostNamesForSuspend(leftHostname, rightHostname, applicationReferences))
                .collect(Collectors.toList());
    }

    private int compareHostNamesForSuspend(HostName leftHostname, HostName rightHostname,
                                           Map<HostName, ApplicationInstanceReference> applicationReferences) {
        ApplicationInstanceReference leftApplicationReference = applicationReferences.get(leftHostname);
        assert leftApplicationReference != null;

        ApplicationInstanceReference rightApplicationReference = applicationReferences.get(rightHostname);
        assert rightApplicationReference != null;

        // ApplicationInstanceReference.toString() is e.g. "hosted-vespa:routing:dev:ci-corp-us-east-1:default"
        int diff = leftApplicationReference.toString().compareTo(rightApplicationReference.toString());
        if (diff != 0) {
            return diff;
        }

        return leftHostname.toString().compareTo(rightHostname.toString());
    }

    private HostStatus getNodeStatus(ApplicationInstanceReference applicationRef, HostName hostName) {
        return statusService.forApplicationInstance(applicationRef).getHostStatus(hostName);
    }

    private void setApplicationStatus(ApplicationId appId, ApplicationInstanceStatus status) 
            throws ApplicationStateChangeDeniedException, ApplicationIdNotFoundException{
        ApplicationInstanceReference appRef = OrchestratorUtil.toApplicationInstanceReference(appId, instanceLookupService);
        try (MutableStatusRegistry statusRegistry =
                     statusService.lockApplicationInstance_forCurrentThreadOnly(appRef)) {

            // Short-circuit if already in wanted state
            if (status == statusRegistry.getApplicationInstanceStatus()) return;

            // Set content clusters for this application in maintenance on suspend
            if (status == ApplicationInstanceStatus.ALLOWED_TO_BE_DOWN) {
                ApplicationInstance<ServiceMonitorStatus> application = getApplicationInstance(appRef);

                // Mark it allowed to be down before we manipulate the clustercontroller
                OrchestratorUtil.getHostsUsedByApplicationInstance(application)
                        .forEach(h -> statusRegistry.setHostState(h, HostStatus.ALLOWED_TO_BE_DOWN));

                // If the clustercontroller throws an error the nodes will be marked as allowed to be down
                // and be set back up on next resume invocation.
                setClusterStateInController(application, ClusterControllerState.MAINTENANCE);
            }

            statusRegistry.setApplicationInstanceStatus(status);
        }
    }

    private void setClusterStateInController(ApplicationInstance<ServiceMonitorStatus> application,
                                             ClusterControllerState state) 
            throws ApplicationStateChangeDeniedException, ApplicationIdNotFoundException {
        // Get all content clusters for this application
        Set<ClusterId> contentClusterIds = application.serviceClusters().stream()
                .filter(VespaModelUtil::isContent)
                .map(ServiceCluster::clusterId)
                .collect(Collectors.toSet());

        // For all content clusters set in maintenance
        log.log(LogLevel.INFO, String.format("Setting content clusters %s for application %s to %s",
                contentClusterIds,application.applicationInstanceId(),state));
        for (ClusterId clusterId : contentClusterIds) {
            ClusterControllerClient client = clusterControllerClientFactory.createClient(
                    VespaModelUtil.getClusterControllerInstancesInOrder(application, clusterId),
                    clusterId.s());
            try {
                ClusterControllerStateResponse response = client.setApplicationState(state);
                if (!response.wasModified) {
                    String msg = String.format("Fail to set application %s, cluster name %s to cluster state %s due to: %s",
                            application.applicationInstanceId(), clusterId, state, response.reason);
                    throw new ApplicationStateChangeDeniedException(msg);
                }
            } catch (IOException e) {
                throw new ApplicationStateChangeDeniedException(e.getMessage());
            }
        }
    }

    private ApplicationInstance<ServiceMonitorStatus> getApplicationInstance(HostName hostName) throws HostNameNotFoundException{
        return instanceLookupService.findInstanceByHost(hostName).orElseThrow(
                () -> new HostNameNotFoundException(hostName));
    }

    private ApplicationInstance<ServiceMonitorStatus> getApplicationInstance(ApplicationInstanceReference appRef) throws ApplicationIdNotFoundException {
        return instanceLookupService.findInstanceById(appRef).orElseThrow(ApplicationIdNotFoundException::new);
    }

    private static void sleep(long time, TimeUnit timeUnit) {
        try {
            Thread.sleep(timeUnit.toMillis(time));
        } catch (InterruptedException e) {
            throw new RuntimeException("Unexpectedly interrupted", e);
        }
    }
}
