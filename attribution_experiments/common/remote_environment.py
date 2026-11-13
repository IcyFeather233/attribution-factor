"""V2 agent semantics over the native Docker transport."""
import logging
from .udocker_environment import UdockerEnvironment, UdockerEnvironmentConfig
from .remote_runtime import RemoteRuntime, RemoteInfrastructureError


class RemoteDockerEnvironment(UdockerEnvironment):
    def __init__(self, *, logger=None, **kwargs):
        self.logger = logger or logging.getLogger(__name__)
        self.config = UdockerEnvironmentConfig(**kwargs)
        self.runtime = RemoteRuntime(self.config)
        self.container_name = self.runtime.container_name
        self.containers_dir = self.runtime.local_dir
        self.scratch_volumes = {}

    def _run_host(self, args, *, timeout, input_text=None, check=False):
        return self.runtime.run(args, timeout=timeout, input_text=input_text, check=check)

    def cleanup(self):
        if getattr(self, 'runtime', None) is not None:
            self.runtime.cleanup()
            self.container_name = ''

    def execute(self, *args, **kwargs):
        result = super().execute(*args, **kwargs)
        if self.runtime.failed:
            raise RemoteInfrastructureError('Remote execution failed; this task must be retried, not graded: '
                                            + str(getattr(self.runtime, 'failure_reason', 'unknown cause')))
        return result

    def serialize(self):
        result = super().serialize()
        result['info']['config']['container_transport'] = 'remote_docker_v1'
        result['info']['config']['remote_host'] = self.runtime.host
        result['info']['config']['image_digest'] = self.runtime.image
        return result
