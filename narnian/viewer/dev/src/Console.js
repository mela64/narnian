import { useEffect, useState, useRef } from "react";
import { callAPI, out } from "./utils";
import { Inbox } from "lucide-react";  // icon

export default function Console({ _agentName_, _isPaused_, _setBusy_}) {
    out("[Console] " +
        "_agentName_: " + _agentName_ + ", " +
        "_isPaused_: " + _isPaused_);

    // all messages shown in the console
    const [messages, setMessages] = useState([]);

    // identifier (time step "k") of the last message that was stored in the message-buffer of this console
    const [lastStoredStep, setLastStoredStep] = useState(-1);
    const lastStoredStepRef = useRef(lastStoredStep);

    // this is a reference to the last shown message, used to force the scrolling operation when a new message is added
    const messagesEndRef = useRef(null);

    // update the reference that is used for automatic scrolling purposes only
    useEffect(() => {
        out("[Console] useEffect *** update reference: messagesEndRef ***");
        messagesEndRef.current?.scrollIntoView({behavior: "smooth", block: "end"});
    }, [messages]); // when new messages are added, this variable changes

    // update reference
    useEffect(() => {
        out("[Console] useEffect *** update reference: lastStoredStepRef ***");
        lastStoredStepRef.current = lastStoredStep;
    }, [lastStoredStep]);

    useEffect(() => {
        if (!_isPaused_) {
            out("[Console] useEffect *** fetching data (get console messages) *** (skipping, not paused)");
            return;
        }

        // this will tell the parent that this component is working
        _setBusy_(true);

        out("[Console] useEffect *** fetching data (agent_name: " + _agentName_ + ") ***");
        callAPI('/get_console', "agent_name=" + _agentName_,
            (x) => {
                // the expected format of the received data, that is a circular buffer that does not start necessarily
                // from zero, is:
                // {"output_messages_last_pos": x, "output_messages_count": y, "output_messages", [...]}

                // here we compute the ID of the first element of the circular buffer
                let posId = x.output_messages_last_pos - x.output_messages_count + 1;
                if (posId < 0)
                    posId = x.output_messages.length + posId;

                // reordering the circular buffer so that the first message has index 0
                const newMessages = [];  // reordered buffer
                let _lastStoredStep = lastStoredStepRef.current
                let isFirst = true;
                for (let i = 0; i < x.output_messages_count; i++) {
                    if (x.output_messages_ids[posId] > _lastStoredStep) {

                        // if there is a "gap" between what was shown and the new batch of messages, we show "..."
                        if (isFirst && x.output_messages_ids[posId] - _lastStoredStep > 1) {
                            isFirst = false;
                            newMessages.push("...");
                        }

                        // saving message in the reordered buffer
                        newMessages.push(x.output_messages[posId]);

                        // updating last step index (we know step indices are ordered)
                        _lastStoredStep = x.output_messages_ids[posId]
                    }
                    posId = (posId + 1) % x.output_messages.length; // next ID in the circular buffer
                }

                // updating state variables
                setLastStoredStep(_lastStoredStep); // last step
                setMessages((prev) => [...prev, ...newMessages]);  // message buffer
            },
            () => setMessages((prev) => (prev)),
            () => {
                // this will tell the parent that this component is now ready
                _setBusy_(false);
            }
        );
    }, [_isPaused_, _agentName_, _setBusy_]);
    // listen to the pause state (_isPaused_), while _agentName_ and _setBusy_ are not going to change

    // returning the <div>...</div> that will be displayed when no messages are there at all (some icon)
    if (messages.length === 0) {
        return (
            <div className="flex items-center justify-center w-full h-full">
                <div className="flex items-center justify-center w-full h-full">
                    <Inbox className="w-12 h-12 text-gray-400"/>
                </div>
            </div>
        );
    }

    // returning the <div>...</div> that will be displayed when the console is actually working
    return (
        <div className="flex items-start justify-start w-full h-full">
            <div className="w-full h-full pb-2 pl-2 pr-2 pt-1">
                <div className="overflow-y-auto h-full">
                    <ul>
                        {messages.map((message, index) => (
                            <li
                                key={index}
                                className={`text-sm text-left p-0 pt-1 break-words ${index === messages.length - 1 ?
                                    'font-bold text-black' : 'text-gray-700'} hover:bg-gray-200 transition`}
                            >
                                {message}
                            </li>
                        ))}
                    </ul>
                    <div ref={messagesEndRef}/>
                    {/* this div ensures automatic scrolling to the bottom */}
                </div>
            </div>
        </div>
    );
}
