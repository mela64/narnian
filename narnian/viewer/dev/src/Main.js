import {useState, useRef, useCallback, useEffect} from "react";  // react
import {DndProvider, useDrag, useDrop} from "react-dnd";  // still about react
import {HTML5Backend} from "react-dnd-html5-backend";  // still about react
import {motion, AnimatePresence} from "framer-motion";  // animated opening panels
import FSM from "./FSM";
import Console from "./Console";
import PlotFigure from "./PlotFigure";
import {callAPI, out} from "./utils";

// icons
import {
    Settings,
    Bot,
    Activity,
    Search,
    Waves,
} from "lucide-react";

let clickTimeout; // timer that triggers a reaction to the click action
let clickCount = 0;  // number of consecutive clicks (to distinguish clicks from double clicks)

// icons used in the agent buttons or stream buttons
const agentButtonIcons = [<Settings/>, <Bot/>];
const streamButtonIcons = [<Activity/>, <Search/>, <Waves/>];


// the structure representing the play/pause status, in its initial (unknown) setting
const unknownPlayPauseStatus = {
    "status": "?",
    "still_to_play": -1
}

export default function Main() {

    // working-state of the components used in this page
    const [isFSMBusy, setIsFSMBusy] = useState(false);
    const [isConsoleBusy, setIsConsoleBusy] = useState(false);
    const [isPlotFigureBusy, setIsPlotFigureBusy] = useState(false);

    // whenever an agent button is clicked, an agent panel is opened (with the FSM, console, streams, stream panels)
    const [agentButtons, setAgentButtons] = useState([]);
    const [openAgentPanels, setOpenAgentPanels] = useState([]);

    // whenever the FSM or console buttons are clicked, they are shown or not
    const [openFSMPanels, setOpenFSMPanels] = useState([]);
    const [openConsolePanels, setOpenConsolePanels] = useState([]);

    // whenever a stream button is clicked, a stream panel is opened (the one with the plot figure)
    const [streamButtons, setStreamButtons] = useState([]);
    const [openStreamPanels, setOpenStreamPanels] = useState({});

    // the column-layout of the agent panels: 1 column when only 1 agent is shown, two columns otherwise
    const [gridCols, setGridCols] = useState("grid-cols-1");

    // the name of the environment (set to "?" if unknown)
    const [envName, setEnvName] = useState("?");

    // the structure with the current play/pause status of the environment (see "unknownPlayPauseStatus" above)
    const [playPauseStatus, setPlayPauseStatus] = useState(unknownPlayPauseStatus);

    // flag that avoids asking the play/pause status more than once, when waiting for a reply from the API
    const playPauseStatusAskedRef = useRef(false);

    // flag that tells if the environment is in pause mode
    const [isPaused, setIsPaused] = useState(false);

    // the value of the selected play option: \u221E (inf), 1S, 1, 100, 1k, 100k
    const [selectedPlayOption, setSelectedPlayOption] = useState("1S");

    // references to "streamButtons" above and "openStreamPanels" above, used in click (and similar) callbacks
    const streamButtonsRef = useRef(streamButtons);
    const agentButtonsRef = useRef(agentButtons);
    const openStreamPanelsRef = useRef(openStreamPanels);

    out("[Main]");

    // keeping references up-to-date at each rendering operation
    useEffect(() => {
        out("[Main] useEffect *** updating streamButtonsRef ***");
        streamButtonsRef.current = streamButtons;
    }, [streamButtons]);

    // keeping references up-to-date at each rendering operation
    useEffect(() => {
        out("[Main] useEffect *** updating openStreamPanelsRef ***");
        openStreamPanelsRef.current = openStreamPanels;
    }, [openStreamPanels]);

    // keeping references up-to-date at each rendering operation
    useEffect(() => {
        out("[Main] useEffect *** updating agentButtonsRef ***");
        agentButtonsRef.current = agentButtons;
    }, [agentButtons]);

    // first thing to do when loading the page: getting the name of the environment
    useEffect(() => {
        out("[Main] useEffect *** fetching data (environment name) ***");

        callAPI('/get_env_name', null,
            (x) => setEnvName(x),
            () => setEnvName("?"),
            () => {
            });
    }, []);

    // getting list of agents
    useEffect(() => {
        if (envName === "?") {
            out("[Main] useEffect *** fetching data (list of agents) *** (skipping, missing env name)");
            return;
        } else {
             out("[Main] useEffect *** fetching data (list of agents) ***")
        }

        callAPI('/get_list_of_agents', "agent_name=" + envName,
            (x) => {
                x.unshift(envName); // adding the name of the environment as extra agent
                const agent_buttons = x.map((label, index) => ({
                    id: index + 1,  // button indices start from 1
                    label,
                    icon: index === 0 ? agentButtonIcons[0] : agentButtonIcons[1],
                }));
                setOpenFSMPanels(agent_buttons.map(agent_button => agent_button.id));
                setOpenConsolePanels(agent_buttons.map(agent_button => agent_button.id));
                setAgentButtons(agent_buttons);
            },
            () => setAgentButtons([]),
            () => {});
    }, [envName]);  // when the status of the loading env name operation changes, we get the agent list

    // when paused, we get the list of streams for all the agents of the environment
    useEffect(() => {
        if (!isPaused) {
            out("[Main] useEffect *** fetching data (list of streams for all agents) *** (skipping, not paused)");
            return;
        }

        out("[Main] useEffect *** fetching data (list of streams for all agents) ***");

        agentButtonsRef.current.forEach((agentButton) => {
            getStreamsAndUpdateStreamButtons(agentButton.label, agentButton.id, streamButtonsRef.current);
        });
    }, [isPaused]);

    // getting the play/pause status of the environment when loading the page the first time
    useEffect(() => {
        if (envName === "?") {
            out("[Main] useEffect *** fetching data (play/pause status) *** (skipping, missing env name)");
            return;
        }

        out("[Main] useEffect *** fetching data (play/pause status) ***");

        getAndUpdatePlayPauseStatus(
            () => {}, // "if playing" callback (do nothing)
            () => {
                setPlayPauseStatus(unknownPlayPauseStatus);
            } // "if error" callback
        );
    }, [envName]);
    // wait for the environment name to be loaded, then ask for play/pause status

    // whenever a new agent panel is opened, the column-layout could switch from 1 to 2 columns
    useEffect(() => {
        out("[Main] useEffect *** possibly changing column layout 1<->2 ***");
        setGridCols(openAgentPanels.length > 1 ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-1 max-w-[1280px]");
    }, [openAgentPanels]);  // listen to changes to the agent panels that are opened

    // open a new agent panel or closes an already opened one
    const toggleAgentPanel = (_agentName_, _agentButtonId_) => {
        setOpenAgentPanels((prev) => {
            if (prev.includes(_agentButtonId_)) {
                return prev.filter((pid) => pid !== _agentButtonId_);
            } else {
                getStreamsAndUpdateStreamButtons(_agentName_, _agentButtonId_, streamButtonsRef.current);  // for all
                return [...prev, _agentButtonId_];
            }
        });
    };

    // open a new stream panel (the one with the plot figure) or closes an already opened one
    // a "stream panel" is just an array with the list of "stream button IDs" that are currently "active"
    const toggleStreamPanel = (_agentButtonId_, _streamButtonId_) => {
        setOpenStreamPanels((prevOpenStreamPanels) => {
            const currentStreamButtonsInCurrentAgentPanel = prevOpenStreamPanels[_agentButtonId_] || [];
            const newStreamButtonsInCurrentAgentPanel =
                currentStreamButtonsInCurrentAgentPanel.includes(_streamButtonId_)
                ? currentStreamButtonsInCurrentAgentPanel.filter((id) => id !== _streamButtonId_)
                : [...currentStreamButtonsInCurrentAgentPanel, _streamButtonId_];
            return {...prevOpenStreamPanels, [_agentButtonId_]: newStreamButtonsInCurrentAgentPanel};
        });
    };

    // open a new FSM (agent-related) panel or closes an already opened one
    const toggleFSMPanel = (_agentButtonId_) => {
        setOpenFSMPanels((prev) => {
            if (prev.includes(_agentButtonId_)) {
                return prev.filter((pid) => pid !== _agentButtonId_);
            } else {
                return [...prev, _agentButtonId_];
            }
        });
    };

    // open a new console (agent-related) panel or closes an already opened one
    const toggleConsolePanel = (_agentButtonId_) => {
        setOpenConsolePanels((prev) => {
            if (prev.includes(_agentButtonId_)) {
                return prev.filter((pid) => pid !== _agentButtonId_);
            } else {
                return [...prev, _agentButtonId_];
            }
        });
    };

    // dropping a stram button on top of another one
    const handleDrop = useCallback((_agentButtonIdOfDropped_, _streamButtonIdOfDropped_,
                                    _agentButtonIdOfTarget_, _streamButtonIdOfTarget_) => {

        // avoid dropping a button on itself or mixing buttons of different agents
        if (_streamButtonIdOfDropped_ === _streamButtonIdOfTarget_) return;
        if (_agentButtonIdOfDropped_ !== _agentButtonIdOfTarget_) return;

        // getting references to the involved players
        const agentButtonId = _agentButtonIdOfDropped_; // same between the two buttons (important!)
        const draggedButton = streamButtonsRef.current[_agentButtonIdOfDropped_]
            .find((sp) => sp.id === _streamButtonIdOfDropped_);
        const targetButton = streamButtonsRef.current[_agentButtonIdOfDropped_]
            .find((sp) => sp.id === _streamButtonIdOfTarget_);

        // merging the lists of IDs/labels of the two buttons (warning: assuming there are no duplicate IDs!)
        // we preserve the order of the IDs, giving priority to the target button IDs
        const mergedIds = [...targetButton.mergedIds, ...draggedButton.mergedIds];
        const mergedLabels = [...targetButton.mergedLabels, ...draggedButton.mergedLabels];

        // creating the new button about the merged streams
        const newStreamButton = {
            id: _streamButtonIdOfTarget_,  // we keep the ID of the target button, that will be replaced by this one
            label: targetButton.label + "+",
            icon: streamButtonIcons[2],
            mergedIds: mergedIds,
            mergedLabels: mergedLabels,
            mergedButtons: [targetButton, draggedButton],
            agentButtonId: agentButtonId
        };

        // closing the "stream panel" of the button that has been dropped on top of another one
        if (openStreamPanelsRef.current[agentButtonId]?.includes(_streamButtonIdOfDropped_)) {
            toggleStreamPanel(agentButtonId, _streamButtonIdOfDropped_);
        }

        // updating the list of stream buttons, removing the dropped button and the target button
        // (that will be replaced by the new merged button, which is using the ID of such target button)
        setStreamButtons((prevStreamButtons) => {
            // filtering out the dragged button
            const curStreamButtons = prevStreamButtons[agentButtonId].filter(
                (btn) =>
                    btn.id !== _streamButtonIdOfTarget_ && // discarding original target button (will be replaced below)
                    btn.id !== _streamButtonIdOfDropped_ // discarding original dragged button
            );

            // adding the new merged button to the already-filtered set of buttons
            return {
                ...prevStreamButtons,
                [agentButtonId]: [...curStreamButtons, newStreamButton], // Add the merged button to the specific panel
            };
        });
    }, []); // we have to list on those variables that will be updated here

    // checks if a stream button is active (i.e., there is an opened stream panel associated to it) or not
    function checkIfActive(_agentButtonId_, _streamButtonId_) {
        return openStreamPanels[_agentButtonId_]?.includes(_streamButtonId_);
    }

    // handle click on a "stream button" (it will open/close the corresponding "stream panel")
    const handleClick = useCallback((_agentButtonIdOfClicked_, _streamButtonIdOfClicked_)=> {

        // this code is just to avoid joinly catching "click" and "double-click" (we pay the penalty of waiting a bit)
        clickCount++;
        if (clickCount === 1) {
            clickTimeout = setTimeout(() => {

                // in case of single-click on a "stream button", we open/close the corresponding panel
                toggleStreamPanel(_agentButtonIdOfClicked_, _streamButtonIdOfClicked_);

                clickCount = 0;
            }, 250);  // here is the penalty we pay before seeing the effects of each click
        } else if (clickCount === 2) {
            clearTimeout(clickTimeout);
            clickCount = 0;
        }
    }, []);

     // handle double-click on a "stream button" (it un-merges merged buttons)
    const handleDoubleClick = useCallback((_agentButtonIdOfClicked_, _streamButtonIdOfClicked_) => {

        // this is to stop the procedure that was distringuishing clicks from double clicks
        clearTimeout(clickTimeout);

        // getting a reference to the double-clicked stream button
        const streamButtonClicked = streamButtonsRef.current[_agentButtonIdOfClicked_]
            .find((sp) => sp.id === _streamButtonIdOfClicked_);

        // if there is nothing to un-merge, stop here
        if (streamButtonClicked.mergedButtons.length === 0) {
            return;
        }

        // preparing data for the un-merge operation
        const restoredButtons = [...streamButtonClicked.mergedButtons]; // copy! (important)

        // if the stream panel of the merged-button was open, we close it
        if (openStreamPanelsRef.current[_agentButtonIdOfClicked_]?.includes(_streamButtonIdOfClicked_)) {
            toggleStreamPanel(_agentButtonIdOfClicked_, _streamButtonIdOfClicked_);
        }

        // updating the list of stream buttons with the newly created single-stream buttons
        // (the merged-button that was turned back into a single-stream button is already there, so we filter it out)
        setStreamButtons((prev) => ({
            ...prev,
            [_agentButtonIdOfClicked_]: [
                ...prev[_agentButtonIdOfClicked_].filter((btn) => btn.id !== streamButtonClicked.id),
                ...restoredButtons, // Add the individual buttons to the specific panel
            ],
        }));
    }, []);

    // downloads the list of streams for a certain agent, and update the list of stream buttons accordingly
    function getStreamsAndUpdateStreamButtons(_agentName_, _agentButtonId_, _streamButtons_) {

        function isGenerated(label) {
            const match = label.match(/generated(\d+)/);
            if (match) {
                return parseInt(match[1], 10);
            }
            return null;
        }

        function isTarget(label) {
            const match = label.match(/target(\d+)/);
            if (match) {
                return parseInt(match[1], 10);
            }
            return null;
        }

        callAPI('/get_list_of_streams', "agent_name=" + _agentName_,
            (x) => {

                // building the list of single stream names that are not on the current buttons (single and merged)
                const missingStreamNames = x.filter(
                    (streamName) =>
                        !_streamButtons_[_agentButtonId_]
                        || !Array.isArray(_streamButtons_[_agentButtonId_])
                        || _streamButtons_[_agentButtonId_].length === 0
                        || !_streamButtons_[_agentButtonId_].some((streamButton) => (
                            (Array.isArray(streamButton.mergedLabels) && streamButton.mergedLabels.includes(streamName))
                        ))
                );

                // finding the largest ID of the existing stream buttons, also looking inside the mergedIds array
                let maxStreamButtonId = 0;
                if (_streamButtons_[_agentButtonId_]
                    && Array.isArray(_streamButtons_[_agentButtonId_])
                    && _streamButtons_[_agentButtonId_].length > 0) {
                    maxStreamButtonId = _streamButtons_[_agentButtonId_]
                        .map((button) => Math.max(...(button.mergedIds || []))) // max value for each mergedIds
                        .reduce((max, current) => Math.max(max, current), -Infinity); // max of all the maxes
                }

                // creating new stream buttons
                const newStreamButtons = missingStreamNames.map((streamName, index) => ({
                    id: index + maxStreamButtonId + 1, // recall that button IDs start from 1
                    label: streamName.slice(0, -4),  // removing " [y]", and " [d]"
                    icon: streamName.endsWith("[y]") ? streamButtonIcons[0] : streamButtonIcons[1],
                    mergedIds: [index + maxStreamButtonId + 1],
                    mergedLabels: [streamName],
                    mergedButtons: [],
                    agentButtonId: _agentButtonId_
                }));

                // altering the just created buttons
                // merging buttons that are about generated data and target data (e.g., "generated1" and "target1")
                const alteredNewStreamButtons = [];
                for (let z = 0; z < newStreamButtons.length; z++) {
                    let isPaired = false;

                    // let's skip buttons that were filtered out (see the end of this loop)
                    if (!newStreamButtons[z]) {
                        continue;
                    }

                    // check stream name: is it a generated/target stream?
                    const generatedNum = isGenerated(newStreamButtons[z].mergedLabels[0]);
                    const targetNum = isTarget(newStreamButtons[z].mergedLabels[0]);

                    // if the name of the stream is "generatedX" or "targetX", we check if we find the paired stream
                    if (generatedNum || targetNum) {

                        // altering case: we need to merge "generatedX" and "targetX", let's search for the other guy
                        const suffix = newStreamButtons[z].mergedLabels[0].slice(-3) // get "[y]" or "[d]"
                        for (let zz = z + 1; zz < newStreamButtons.length; zz++) {

                            // let's skip buttons that were filtered out (see the end of this loop)
                            if (!newStreamButtons[zz]) {
                                continue;
                            }

                            // excluding not-matching "[y]" or "[d]"
                            if (!newStreamButtons[zz].mergedLabels[0].endsWith(suffix)) {
                                continue;
                            }

                            // looking for the other stream of the pair
                            if (generatedNum && generatedNum >= 0) {

                                // given "generatedX", we want a target that ends with the same "X", and vice-versa
                                if (generatedNum !== isTarget(newStreamButtons[zz].mergedLabels[0])) {
                                    continue;
                                }
                            } else {

                                // given "generatedX", we want a target that ends with the same "X", and vice-versa
                                if (targetNum !== isGenerated(newStreamButtons[zz].mergedLabels[0])) {
                                    continue;
                                }
                            }

                            const generatedZ = (generatedNum && generatedNum >= 0) ? z : zz;
                            const targetZ = (generatedNum && generatedNum >= 0) ? zz : z;

                            // if a pair "generatedX" and "targetX" was found... merge!
                            const mergedIds =
                                [...newStreamButtons[generatedZ].mergedIds,
                                    ...newStreamButtons[targetZ].mergedIds];
                            const mergedLabels =
                                [...newStreamButtons[generatedZ].mergedLabels,
                                    ...newStreamButtons[targetZ].mergedLabels];

                            // creating the new button about the merged streams
                            const newStreamButton = {
                                id: newStreamButtons[generatedZ].id,
                                label: newStreamButtons[generatedZ].label,
                                icon: newStreamButtons[generatedZ].icon,
                                mergedIds: mergedIds,
                                mergedLabels: mergedLabels,
                                mergedButtons: [],
                                agentButtonId: newStreamButtons[generatedZ].agentButtonId
                            };

                            // saving
                            alteredNewStreamButtons.push(newStreamButton);

                            // let's avoid looking again for this button in the original array
                            newStreamButtons[generatedZ] = null;
                            newStreamButtons[targetZ] = null;
                            isPaired = true;
                            break; // stop searching
                        }
                    }

                    if (!isPaired) {

                        // simple case: nothing to alter, just get the button as it is
                        alteredNewStreamButtons.push(newStreamButtons[z]);
                    }
                }

                // updating the current buttons with the newly created ones
                setStreamButtons((prevStreamButtons) => ({
                    ...prevStreamButtons,
                    [_agentButtonId_]: [...(prevStreamButtons[_agentButtonId_] || []), ...alteredNewStreamButtons],
                }));
            },
            () => setStreamButtons((prev) => (prev)),
            () => {
            });
    }

    // start the timer associated to the play button
    const startOrStopPlayTimer = () => {
        let delay = 100; // Initial delay
        let maxDelay = 1000; // Maximum delay (after which it stays fixed)

        function timerTriggeredFcn() {
            getAndUpdatePlayPauseStatus(() =>
            {
                // "if playing" callback (if it is "paused" this will not be called, so it will stop)
                delay = Math.min(delay * 2, maxDelay);  // increase delay
                setTimeout(timerTriggeredFcn, delay);  // try again
            }, () =>
            {
                // "if error" callback
                //setTimeout(timerTriggeredFcn, delay);  // try again
            });
        }

        // start the first iteration
        setTimeout(timerTriggeredFcn, delay);
    }

    // get the current status of play/pause, and calls two optional callback
    function getAndUpdatePlayPauseStatus(_playCallback_, _errorCallback_) {
        if (!playPauseStatusAskedRef.current) {
            setIsPaused(false);
            playPauseStatusAskedRef.current = true;

            callAPI('/get_play_pause_status', null,
                (x) => {
                    if (x.status === "playing") {
                        if (_playCallback_) {
                            _playCallback_();
                        }
                        setIsPaused(false);
                    } else if (x.status === "paused") {
                        setIsPaused(true);
                    } else if (x.status === "ended") {
                        setIsPaused(true);
                    } else {
                        throw new Error("Unknown status: " + x.status);
                    }
                    setPlayPauseStatus(x);
                },
                () => {
                    if (_errorCallback_) {
                        _errorCallback_();
                    }
                },
                () => {
                    playPauseStatusAskedRef.current = false;
                });
        }
    }

    // handle the click on the play/pause button
    const handleClickOnPlayPauseButton = () => {

        // if something is drawing/working/fetching-data, do not let it go
        if (isFSMBusy || isConsoleBusy || isPlotFigureBusy) {
            out("[Main] *** click on play/pause button *** (ignored due to other components busy)");
            return;
        } else {
            out("[Main] *** click on play/pause button ***");
        }

        if (playPauseStatus.status === "paused") {

            // getting play options (number of steps to run)
            const steps = selectedPlayOption.endsWith("k") ?
                parseInt(selectedPlayOption.replace('k', '')) * 1000 :
                selectedPlayOption === "1S" ? -1 : selectedPlayOption === "\u221E" ? -2 : parseInt(selectedPlayOption)

            // asking to play
            out("[Main] *** fetching data (ask-to-play for " + steps + " steps) ***");
            callAPI('/ask_to_play', "steps=" + steps,
                (x) => {
                    out("[Main] server responded to the ask-to-play request, " +
                        "which was received by the sever when at step id: " + x);
                    startOrStopPlayTimer();  // starting timer!
                },
                () => {
                },
                () => {
                });

        } else if (playPauseStatus.status === "playing") {

            // asking to pause
            out("[Main] *** fetching data (ask-to-pause) ***");
            callAPI('/ask_to_pause', null,
                (x) => {
                    out("[Main] server responded to the ask-to-pause request, " +
                        "which was received by the sever when at step id: " + x);
                    startOrStopPlayTimer(); // stop timer!
                },
                () => {
                },
                () => {
                });
        } else {
            // do nothing
        }
    };

    // returning what will be displayed in the "root" <div>...</div>
    return (
        <DndProvider backend={HTML5Backend}>

            <div className="p-6 space-y-8 flex flex-col items-center w-full">
                <div className="flex flex-col items-center justify-center text-center">
                    <h1 className="text-2xl font-semibold mt-2">NARNIAN</h1>
                    <h1 className="text-2xl font-semibold mt-2">Environment:{" "}
                        {envName}</h1>
                </div>

                <div className="flex flex-wrap gap-4 w-full justify-center">

                    <div className="flex items-center"
                         style={{ display: playPauseStatus.status === 'ended' ? 'none' : 'flex' }}>

                        <span className="text-xs font-semibold w-20 text-right block mr-2">
                          {playPauseStatus.status === '?' ? "What?" :
                              ((playPauseStatus.status === 'playing'
                              && playPauseStatus.still_to_play > 1) ?
                                  playPauseStatus.still_to_play : "")}
                        </span>

                        <button
                            className={`relative flex items-center justify-center mr-1 pointer-events-none 
                            h-6 w-6 rounded-full 
                            ${playPauseStatus.status === 'paused' ? "bg-green-500" :
                                    (playPauseStatus.status === 'ended' ? "bg-gray-400" :
                                            (playPauseStatus.status === 'playing' ?
                                                    (playPauseStatus.still_to_play === 1 ? "bg-orange-400" :
                                                            (playPauseStatus.still_to_play !== 0 ? "bg-red-500" :
                                                                   "bg-blue-400" // "unexpected value still_to_play!"
                                                            )
                                                    ) :  "bg-gray-400" // "unexpected status!"
                                            )
                                    )
                            }`}
                        >
                        </button>

                    </div>

                    <button onClick={handleClickOnPlayPauseButton}
                            className={`px-4 py-2 rounded-2xl bg-amber-200 
                            ${(isFSMBusy || isConsoleBusy || isPlotFigureBusy) ? 
                                "hover:bg-gray-200" : "hover:bg-amber-300"}`}
                            style={{ display: playPauseStatus.status === 'ended' ? 'none' : 'flex' }} >

                        {playPauseStatus.status === 'playing' ? (

                            // pause icon
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24"
                                 stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round"
                                                             strokeWidth="2" d="M6 19V5M18 19V5"/>
                            </svg>

                        ) : (playPauseStatus.status === 'paused' ? (

                            // play icon
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24"
                                 stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round"
                                                             strokeWidth="2" d="M10 19V5l12 7-12 7z"
                                                             transform="translate(-2, 0)"/>
                            </svg>
                        ) : (

                            // question mark icon
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M9.09 9a3 3 0 115.82 0c0 2-3 2-3 4v1"/>
                                <circle cx="12" cy="17" r="0.75" strokeWidth="1.5"/>
                            </svg>

                        ))}

                    </button>

                    <div className="flex gap-2 items-center"
                         style={{ display: playPauseStatus.status === 'ended' ? 'none' : 'flex' }}>
                        {["1S", "1", "100", "1k", "100k", "\u221E"].map((option) => (
                            <button key={option} onClick={() => setSelectedPlayOption(option)}
                                className={selectedPlayOption === option ? "h-6 text-sm bg-amber-200 hover:bg-amber-300 " +
                                    "px-2 py-0 rounded-2xl" : "h-6 text-sm bg-gray-100 hover:bg-gray-200 px-2 py-0 " +
                                    "rounded-2xl"}>
                                {option}
                            </button>
                        ))}
                    </div>

                    {agentButtons.map((agent_button) => (
                        <button
                            key={agent_button.id}
                            onClick={() => toggleAgentPanel(agent_button.label, agent_button.id)}
                            className={`flex items-center space-x-2 px-4 py-2 rounded-2xl shadow-md transition-colors ${
                                openAgentPanels.includes(agent_button.id)
                                    ? "bg-blue-500 text-white"
                                    : "bg-gray-100 hover:bg-gray-200"
                            }`}>
                            {agent_button.icon}<span>{agent_button.label}</span>
                        </button>
                    ))}
                </div>

                <div className={`grid ${gridCols} gap-8 w-full`}>
                    {agentButtons.map(
                        (agent_button) =>
                            openAgentPanels.includes(agent_button.id) &&
                            (  // **** big thing opening here.... ***
                                <AnimatePresence key={agent_button.id}>
                                    <motion.div initial={{opacity: 0, height: 0}}
                                                animate={{opacity: 1, height: "auto"}}
                                                exit={{opacity: 0, height: 0}}
                                                transition={{duration: 0.3}}
                                                className="w-full p-4 bg-white rounded-2xl shadow-lg border space-y-6">

                                        <h2 className="font-medium text-lg flex items-center justify-center">
                                            <span className="mr-1">{agent_button.label}</span>
                                            <button
                                                className={`w-6 h-6 
                                                ${openFSMPanels.includes(agent_button.id) ? 
                                                    "text-white bg-blue-500" : "bg-gray-100"} rounded-full 
                                                    flex items-center justify-center ml-2`}
                                                onClick={() => toggleFSMPanel(agent_button.id)}>
                                                B
                                            </button>
                                            <button
                                                className={`w-6 h-6 
                                                ${openConsolePanels.includes(agent_button.id) ? 
                                                    "text-white bg-blue-500" : "bg-gray-100"} rounded-full flex 
                                                    items-center justify-center ml-2`}
                                                onClick={() => toggleConsolePanel(agent_button.id)}>
                                                C
                                            </button>
                                        </h2>

                                        <div className={`grid grid-cols-1 ${(openFSMPanels.includes(agent_button.id) &&
                                            openConsolePanels.includes(agent_button.id)) ?
                                            "sm:grid-cols-2" : "sm:grid-cols-1"} gap-4`}>

                                            {openFSMPanels.includes(agent_button.id) &&
                                                <div className="h-[400px] w-full flex justify-center">
                                                <div className="max-w-[500px] w-full p-0 pt-4 pb-5 bg-gray-100
                                                        rounded-xl shadow text-center">
                                                        <h3 className="font-medium">Behaviour</h3>
                                                        <FSM _agentName_={agent_button.label}
                                                             _isPaused_={isPaused}
                                                             _setBusy_={setIsFSMBusy}
                                                        />
                                                    </div>
                                                </div>
                                            }

                                            {openConsolePanels.includes(agent_button.id) &&
                                                <div className="h-[400px] w-full flex justify-center">
                                                    <div className="max-w-[500px] w-full p-0 pt-4 pb-5 bg-gray-100
                                                        rounded-xl shadow text-center">
                                                        <h3 className="font-medium">Console</h3>
                                                        <Console _agentName_={agent_button.label}
                                                                 _isPaused_={isPaused}
                                                                 _setBusy_={setIsConsoleBusy}
                                                        />
                                                    </div>
                                                </div>
                                            }

                                        </div>

                                        <div className="flex gap-4 justify-center w-full flex-wrap">
                                            {streamButtons[agent_button.id]?.map((streamButton) => (
                                                <AnimatePresence key={streamButton.mergedIds.join("-")}>
                                                    <motion.div initial={{opacity: 0, scale: 0.9}}
                                                                animate={{opacity: 1, scale: 1}}
                                                                exit={{opacity: 0, scale: 0.9}}
                                                                transition={{duration: 0.2}}>

                                                        <DraggableStreamButton
                                                            _streamButton_={streamButton}
                                                            _onDrop_={(droppedStreamButton) =>
                                                                handleDrop(
                                                                    droppedStreamButton.agentButtonId,
                                                                    droppedStreamButton.id,
                                                                    streamButton.agentButtonId,
                                                                    streamButton.id)}
                                                            _onDoubleClick_={() =>
                                                                handleDoubleClick(
                                                                    streamButton.agentButtonId,
                                                                    streamButton.id)}
                                                            _onClick_={() =>
                                                                handleClick(
                                                                    streamButton.agentButtonId,
                                                                    streamButton.id)}
                                                            _checkIfActive_={() =>
                                                                checkIfActive(
                                                                    streamButton.agentButtonId,
                                                                    streamButton.id)}
                                                        />
                                                    </motion.div>
                                                </AnimatePresence>
                                            ))}
                                        </div>

                                        <div className={`gap-4 mt-6r  
                                        ${openStreamPanels[agent_button.id]?.length <= 2 ?
                                            (openStreamPanels[agent_button.id]?.length <= 1 ? 
                                                " grid sm:grid-cols-1 max-w-[900px] mx-auto" 
                                                : "grid sm:grid-cols-2") : "grid sm:grid-cols-3"}`}>
                                            {openStreamPanels[agent_button.id]?.map((id) => {
                                                const streamButton = streamButtons[agent_button.id]?.find(
                                                    (btn) => btn.id === id
                                                );
                                                return (
                                                    <div key={id}
                                                         className="min-h-[500px] p-0 pt-4 pb-8 bg-gray-50 border
                                                            rounded-xl shadow text-center">
                                                        <h3 className="font-medium flex items-center justify-center">
                                                            <span className="w-5 h-5">{streamButton?.icon}</span>
                                                            <span className="ml-1">{streamButton?.label}</span>
                                                        </h3>
                                                        <PlotFigure _agentName_={agent_button.label}
                                                                    _streamStruct_={streamButton}
                                                                    _isPaused_={isPaused}
                                                                    _setBusy_={setIsPlotFigureBusy}
                                                        />
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </motion.div>
                                </AnimatePresence>
                            )  // **** big thing closing here.... ***
                    )}

                </div>
            </div>
        </DndProvider>
    );
}

// create a new stream button with several action handlers attached (given a streamButton structure)
// it is interpreted and built as a React component: keep name with the starting capital letter (otherwise it fails!)
function DraggableStreamButton({_streamButton_, _onDrop_, _onDoubleClick_, _onClick_, _checkIfActive_}) {

    // drag action triggered
    const [{isDragging}, drag] = useDrag(() => ({
        type: "button",  // type of object that is being dragged
        item: _streamButton_,  // stream-button structure
        collect: (monitor) =>
            ({isDragging: monitor.isDragging()}),  // check
    }));

    // drop action triggered
    const [, drop] = useDrop(() => ({
        accept: "button",
        drop: (draggedItem) => _onDrop_(draggedItem),
    }));

    // returning the <div>...</div> that will be displayed to represent the stream button
    return (
        <div
            className={`flex h-6 items-center justify-center px-3 py-4 rounded-2xl shadow-md select-none cursor-move 
            text-center transition-colors whitespace-nowrap ${ 
                _checkIfActive_() ? "bg-blue-600 text-white" : "bg-gray-100 hover:bg-gray-200"
            } ${isDragging ? "opacity-50" : "opacity-100"}`}
            onClick={() => _onClick_(_streamButton_)}
            onDoubleClick={() => _onDoubleClick_(_streamButton_)}
            ref={(node) => { if (node) drag(drop(node)); }}
        >
            <span className="w-5 h-5 flex items-center justify-center">{_streamButton_.icon}</span>
            <span className="ml-1">{_streamButton_.label}</span>
        </div>
    );
}
