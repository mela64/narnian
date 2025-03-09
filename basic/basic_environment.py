from narnian.agent import Agent
from narnian.environment import Environment
from narnian.streams import Stream


class BasicEnvironment(Environment):

    def __init__(self, name: str, title: str | None = None):
        """Create a new basic environment."""

        super(BasicEnvironment, self).__init__(name, title)
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

    def add_agent(self, agent: Agent):
        """Add an agent and mark those action that will be triggered by other agents."""

        super().add_agent(agent)

        # send (don't wait), get (wait), got (wait)
        agent.behav.wait_for_all_actions_that_start_with("get_")
        agent.behav.wait_for_all_actions_that_start_with("got_")

        # ask (don't wait), do (wait), done (wait)
        agent.behav.wait_for_all_actions_that_start_with("do_")
        agent.behav.wait_for_all_actions_that_start_with("done_")

    def add_stream(self, stream: Stream):
        """Add a stream and create a unique, long, set of merged descriptor component labels (attribute labels)."""

        super().add_stream(stream)

        # merging descriptor labels (attribute labels) and sharing them with all streams
        self.share_attributes()
