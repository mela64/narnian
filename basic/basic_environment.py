from narnian.agent import Agent
from narnian.environment import Environment


class BasicEnvironment(Environment):

    def __init__(self, name):
        """Create a new basic environment."""

        super(BasicEnvironment, self).__init__(name)
        self.commands_to_send.append("kill")
        self.commands_to_receive.append("kill")

    def send_command(self, command: str, dest_agent: Agent, data: dict | None = None) -> bool:
        """Send a predefined command to an agent."""

        ret = super().send_command(command, dest_agent, data)

        if command == "kill":
            return dest_agent.receive_command(command, src_agent=None)
        else:
            return ret

    def receive_command(self, command: str, src_agent: Agent, data: dict | None = None) -> bool:
        """Receive a predefined command."""

        ret = super().receive_command(command, src_agent, data)

        if command == "kill":
            self.err(f"In the current implementation, environment cannot be closed.")
            return False
        else:
            return ret
