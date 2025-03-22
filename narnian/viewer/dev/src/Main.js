import {useState, useRef, useCallback, useEffect} from "react";  // react
import {DndProvider, useDrag, useDrop} from "react-dnd";  // still about react
import {HTML5Backend} from "react-dnd-html5-backend";  // still about react
import {motion, AnimatePresence} from "framer-motion";  // animated opening panels
import FSM from "./FSM";
import Console from "./Console";
import PlotFigure from "./PlotFigure";
import Balloon from "./Balloon";
import {callAPI, out, showError} from "./utils";

// icons
import {
    Settings,
    Bot,
    Activity,
    Search,
    Waves,
    GraduationCap,
    Save,
    Upload,
    Download,
    ChevronLeft,
    ChevronRight,
    Hand
} from "lucide-react";

let clickTimeout; // timer that triggers a reaction to the click action
let clickCount = 0;  // number of consecutive clicks (to distinguish clicks from double clicks)

// icons used in the agent buttons or stream buttons
const agentButtonIcons = [<Settings/>, <Bot/>, <GraduationCap/>];
const streamButtonIcons = [<Activity/>, <Search/>, <Waves/>];

// the structure representing the play/pause status, in its initial (unknown) setting
const unknownPlayPauseStatus = {
    "status": "?",
    "still_to_play": -1,
    "time": 0.,
    "y_range": null,
    "matched_checkpoint_to_show": null,
    "more_checkpoints_available": false
}

// the structure representing the ended status
const endedStatus = {
    "status": "ended",
    "still_to_play": -1,
    "time": 0.,
    "y_range": null,
    "matched_checkpoint_to_show": null,
    "more_checkpoints_available": false
}

export default function Main() {

    // working-state of the components used in this page
    const [isBusy, setIsBusy] = useState(0);
    const isBusyRef = useRef(0);

    // whenever an agent button is clicked, an agent panel is opened (with the FSM, console, streams, stream panels)
    const [agentButtons, setAgentButtons] = useState([]);
    const [openAgentPanels, setOpenAgentPanels] = useState([]);
    const openAgentPanelsRef = useRef(openAgentPanels)

    // whenever the FSM or console buttons are clicked, they are shown or not
    const [openFSMPanels, setOpenFSMPanels] = useState([]);
    const openFSMPanelsRef = useRef(openFSMPanels);
    const [openConsolePanels, setOpenConsolePanels] = useState([]);
    const openConsolePanelsRef = useRef(openConsolePanels);
    const [shownOwnedStreams, setShownOwnedStreams] = useState([]);
    const [shownSignals, setShownSignals] = useState([]);
    const [shownDescriptors, setShownDescriptors] = useState([]);
    const shownOwnedStreamsRef = useRef(shownOwnedStreams);
    const shownSignalsRef = useRef(shownSignals);
    const shownDescriptorsRef = useRef(shownDescriptors);
    const showAllRef = useRef(shownSignals)

    // whenever a stream button is clicked, a stream panel is opened (the one with the plot figure)
    const [streamButtons, setStreamButtons] = useState([]);
    const [openStreamPanels, setOpenStreamPanels] = useState({});
    const [visibleStreamButtons, setVisibleStreamButtons] = useState([]);

    // the column-layout of the agent panels: 1 column when only 1 agent is shown, two columns otherwise
    const [gridCols, setGridCols] = useState("grid-cols-1");

    // the name of the environment, set to "?" when unknown
    const [envTitle, setEnvTitle] = useState("?");
    const envTitleRef = useRef(envTitle);
    const envNameRef = useRef("?");

    // the structure with the current play/pause status of the environment (see "unknownPlayPauseStatus" above)
    const [playPauseStatus, setPlayPauseStatus] = useState(unknownPlayPauseStatus);
    const playPauseStatusRef = useRef(playPauseStatus);

    // flag that avoids asking the play/pause status more than once, when waiting for a reply from the API
    const playPauseStatusAskedRef = useRef(false);

    // flag that tells if the environment is in pause mode
    const [isPaused, setIsPaused] = useState(false);

    // the value of the selected play option: \u221E (inf), 1S, 1, 100, 1k, 100k
    const [selectedPlayOption, setSelectedPlayOption] = useState("1S");
    const serverCommunicatedPlayStepsRef = useRef(-1)

    // references to "streamButtons" above and "openStreamPanels" above, used in click (and similar) callbacks
    const streamButtonsRef = useRef(streamButtons);
    const agentButtonsRef = useRef(agentButtons);
    const openStreamPanelsRef = useRef(openStreamPanels);

    // up-lifted data from the plot figures (array of arrays of integers)
    const ioDataRef = useRef(null)

    // saving flag
    const [saving, setSaving] = useState(false);

    // if offline
    const [offline, setOffline] = useState(false);
    const offlineRef = useRef(offline);
    const fileInputRef = useRef(null);

    const firstCheckpointWasAlreadyFoundRef = useRef(false);

    // background colors of the agent panels
    const bgColors = ['bg-white', 'bg-blue-50', 'bg-green-50', 'bg-yellow-50', 'bg-red-50',
        'bg-indigo-50', 'bg-purple-50'];

    out("[Main]");

    // keeping references up-to-date at each rendering operation (this is fine here)
    useEffect(() => {
        out("[Main] useEffect *** updating openStreamPanelsRef ***");
        openStreamPanelsRef.current = openStreamPanels;
    }, [openStreamPanels]);

    useEffect(() => {
        if (saving)
            document.body.classList.add("saving");
        else
            document.body.classList.remove("saving");
    }, [saving]);

    // first thing to do when loading the page: getting the name of the environment
    useEffect(() => {
        out("[Main] useEffect *** fetching data (environment name) ***");

        callAPI('/get_env_name', null,
            (x) => { envNameRef.current = x.name; setEnvTitle(() => {
                envTitleRef.current = x.title; return x.title; }); },
            () => { envNameRef.current = "?"; setEnvTitle(() =>{
                envTitleRef.current = "Offline"; return "Offline"; });
            setOffline(true); return false; },
            () => {
            });
    }, []);

    // getting list of agents
    useEffect(() => {
        if (envNameRef.current === "?") {
            out("[Main] useEffect *** fetching data (list of agents) *** (skipping, missing env name)");
            return;
        } else {
             out("[Main] useEffect *** fetching data (list of agents) ***")
        }

        callAPI('/get_list_of_agents', "agent_name=" + envNameRef.current,
            (x) => {
                const agent_names = x.agents;
                const agent_authorities = x.authorities;

                const firstTime = agentButtonsRef.current.length === 0;
                const needToRefreshButtons = agent_names.some((name, i) =>
                    agentButtonsRef.current.some(
                        button => button.label === name && button.authority !== agent_authorities[i])
                );

                // if it is the first time, or if the authority changed...
                if (firstTime || needToRefreshButtons) {
                    agent_names.unshift(envNameRef.current); // adding the name of the environment as extra agent
                    agent_authorities.unshift(-1.); // fake authority for the environment
                    const agent_buttons = agent_names.map((label, index) => ({
                        id: index + 1,  // button indices start from 1
                        label,
                        authority: agent_authorities[index],
                        icon: index === 0 ? agentButtonIcons[0] : (
                            agent_authorities[index] < 1.0 ? agentButtonIcons[1] : agentButtonIcons[2]),
                    }));

                    // saving a placeholder that includes all agent button IDs
                    showAllRef.current = agent_buttons.map(agent_button => agent_button.id);

                    // if it is the first time (only!), we collect the button IDs in the other panel-associated lists
                    if (firstTime) {
                        setOpenFSMPanels(() => {
                            openFSMPanelsRef.current = showAllRef.current;
                            return showAllRef.current;
                        });
                        setOpenConsolePanels(() => {
                            openConsolePanelsRef.current = showAllRef.current;
                            return showAllRef.current;
                        });
                        setShownSignals(() => {
                            shownSignalsRef.current = showAllRef.current;
                            return showAllRef.current;
                        });
                        setShownDescriptors(() => {
                            shownDescriptorsRef.current = showAllRef.current;
                            return showAllRef.current;
                        });
                        setShownOwnedStreams(() => {
                            shownOwnedStreamsRef.current = [];
                            return [];
                        });
                    }

                    setAgentButtons(() => {
                        agentButtonsRef.current = agent_buttons
                        return agent_buttons;
                    });
                }
            },
            () => {
                if (!offlineRef.current) {
                    setAgentButtons(() => {agentButtonsRef.current = []; return [];}); return true;
                } else { return false; }},
            () => {});
    }, [envTitle, isPaused]);  // when the status of the loading env name operation changes, we get the agent list
    // it the authority changed, we need to update button graphics, so we also call this API when isPaused

    // getting the play/pause status of the environment when loading the page the first time
    useEffect(() => {
        if (envNameRef.current === "?") {
            out("[Main] useEffect *** fetching data (play/pause status) *** (skipping, missing env name)");
            return;
        }

        out("[Main] useEffect *** fetching data (play/pause status) ***");

        getAndUpdatePlayPauseStatus(
            () => {}, // "if playing" callback (do nothing)
            () => {
                if (!offlineRef.current) {
                    setPlayPauseStatus(() => {
                        playPauseStatusRef.current = unknownPlayPauseStatus;
                        return unknownPlayPauseStatus;
                    });
                } else {
                    setIsPaused(() => {
                        setPlayPauseStatus(() => {
                            playPauseStatusRef.current = endedStatus;
                            return endedStatus;
                        });
                        return true;
                    });
                }
            } // "if error" callback
        );
    }, [envTitle]);
    // wait for the environment name to be loaded, then ask for play/pause status

    // in case of offline setting
    useEffect(() => {
        offlineRef.current = offline;
        if (!offline) {
            out("[Main] useEffect *** checking if offline and updating ref (not offline) ***");
        } else {
            out("[Main] useEffect *** checking if offline and updating ref (confirmed: offline) ***");
        }
    }, [offline]);

    // whenever a new agent panel is opened, the column-layout could switch from 1 to 2 columns
    useEffect(() => {
        out("[Main] useEffect *** possibly changing column layout 1<->2 ***");
        setGridCols(openAgentPanels.filter((num) => num > 0).length > 1 ?
            "grid-cols-1 sm:grid-cols-2" : "grid-cols-1 max-w-[1280px]");
    }, [openAgentPanels]);  // listen to changes to the agent panels that are opened

    // open a new agent panel or closes an already opened one
    const toggleAgentPanel = (_agentName_, _agentButtonId_) => {
        setOpenAgentPanels((prev) => {
            if (prev.includes(_agentButtonId_)) {
                const panels = prev.filter((pid) => pid !== _agentButtonId_);
                openAgentPanelsRef.current = panels;
                return panels;
            } else {
                getStreamsAndUpdateStreamButtons(_agentName_, _agentButtonId_);  // for all
                const panels = [...prev, _agentButtonId_];
                openAgentPanelsRef.current = panels;
                return panels;
            }
        });
    };

    const toggleShowOwnedStreams = (_agentName_, _agentButtonId_) => {
        setShownOwnedStreams((prev) => {
            if (!prev.includes(_agentButtonId_)) {
                const newData = [...prev, _agentButtonId_];
                filterStreamButtons(_agentName_, _agentButtonId_,
                    newData, shownSignalsRef.current, shownDescriptorsRef.current);
                shownOwnedStreamsRef.current = newData;
                return newData;
            } else {
                const newData = prev.filter((pid) => pid !== _agentButtonId_);
                filterStreamButtons(_agentName_, _agentButtonId_,
                    newData, shownSignalsRef.current, shownDescriptorsRef.current);
                shownOwnedStreamsRef.current = newData;
                return newData;
            }
        });
    };

    const toggleShowSignals = (_agentName_, _agentButtonId_) => {
        setShownSignals((prev) => {
            if (!prev.includes(_agentButtonId_)) {
                const newData = [...prev, _agentButtonId_];
                filterStreamButtons(_agentName_, _agentButtonId_,
                    shownOwnedStreamsRef.current, newData, shownDescriptorsRef.current);
                shownSignalsRef.current = newData;
                return newData;
            } else {
                const newData = prev.filter((pid) => pid !== _agentButtonId_);
                filterStreamButtons(_agentName_, _agentButtonId_,
                    shownOwnedStreamsRef.current, newData, shownDescriptorsRef.current);
                shownSignalsRef.current = newData;
                return newData;
            }
        });
    };

    const toggleShowDescriptors = (_agentName_, _agentButtonId_) => {
        setShownDescriptors((prev) => {
            if (!prev.includes(_agentButtonId_)) {
                const newData = [...prev, _agentButtonId_];
                filterStreamButtons(_agentName_, _agentButtonId_,
                    shownOwnedStreamsRef.current, shownSignalsRef.current, newData);
                shownDescriptorsRef.current = newData;
                return newData;
            } else {
                const newData = prev.filter((pid) => pid !== _agentButtonId_);
                filterStreamButtons(_agentName_, _agentButtonId_,
                    shownOwnedStreamsRef.current, shownSignalsRef.current, newData);
                shownDescriptorsRef.current = newData;
                return newData;
            }
        });
    };

    const filterStreamButtons = useCallback((_agentName_, _agentButtonId_,
                                             _shownOwnedStreams_, _shownSignals_, _shownDescriptors_) => {
        const keptButtons1 = streamButtonsRef.current[_agentButtonId_].filter((button) =>
            !_shownOwnedStreams_.includes(_agentButtonId_) ||
            (_shownOwnedStreams_.includes(_agentButtonId_)
                && button.mergedLabels[0].toLowerCase().startsWith(_agentName_.toLowerCase()))
        );
        const keptButtons2 = keptButtons1.filter((button) =>
            _shownSignals_.includes(_agentButtonId_) ||
            (!_shownSignals_.includes(_agentButtonId_) && button.mergedLabels[0].endsWith("[d]"))
        );
        const keptButtons3 = keptButtons2.filter((button) =>
            _shownDescriptors_.includes(_agentButtonId_) ||
            (!_shownDescriptors_.includes(_agentButtonId_) && button.mergedLabels[0].endsWith("[y]"))
        );
        setVisibleStreamButtons((prev) => {
            return {
                ...prev,
                [_agentButtonId_]: keptButtons3.map(button => button.id)
            };
        });

        // close the stream panel, if opened
        const discardedButtons = streamButtonsRef.current[_agentButtonId_]
            .filter((button) => !keptButtons3.includes(button))
        discardedButtons.forEach(button => {
            if (openStreamPanelsRef.current[_agentButtonId_]?.includes(button.id)) {
                toggleStreamPanel(_agentButtonId_, button.id);
            }
        });
    }, []);

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
                const panels = prev.filter((pid) => pid !== _agentButtonId_);
                openFSMPanelsRef.current = panels;
                return panels;
            } else {
                const panels = [...prev, _agentButtonId_];
                openFSMPanelsRef.current = panels;
                return panels;
            }
        });
    };

    // open a new console (agent-related) panel or closes an already opened one
    const toggleConsolePanel = (_agentButtonId_) => {
        setOpenConsolePanels((prev) => {
            if (prev.includes(_agentButtonId_)) {
                const panels = prev.filter((pid) => pid !== _agentButtonId_);
                openConsolePanelsRef.current = panels;
                return panels;
            } else {
                const panels = [...prev, _agentButtonId_];
                openConsolePanelsRef.current = panels;
                return panels;
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
            const updatedStreamButtons = {
                ...prevStreamButtons,
                [agentButtonId]: [...curStreamButtons, newStreamButton], // add the merged button to the specific panel
            }

            // refreshing reference
            streamButtonsRef.current = updatedStreamButtons;
            return updatedStreamButtons;
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

        // this is to stop the procedure that was distinguishing clicks from double clicks
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
        setStreamButtons((prev) => {
            const updatedStreamButtons = {
                ...prev,
                [_agentButtonIdOfClicked_]: [
                    ...prev[_agentButtonIdOfClicked_].filter((btn) => btn.id !== streamButtonClicked.id),
                    ...restoredButtons, // Add the individual buttons to the specific panel
                ]
            };

            // refreshing reference
            streamButtonsRef.current = updatedStreamButtons;
            return updatedStreamButtons;
        });
    }, []);

    useEffect(() => {
        if (saving && ioDataRef.current !== null) {
            out("[Main] useEffect *** continuing a save procedure ***");

            const waitingSaveToComplete = () => {
                const shouldStop = doSave();
                if (shouldStop) {
                    setSaving(false);
                } else {
                    setTimeout(() => {
                        waitingSaveToComplete();
                    }, 1500);
                }
            }

            const saveToFile = (data, filename) => {
                const json = JSON.stringify(data);
                const blob = new Blob([json], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            };

            const doSave = () => {

                // counting the total streams that were collected so far ("null" means "not-received-yet")
                const totalReceivedStreams = ioDataRef.current.reduce(
                    (total, subArray) => total +
                        (subArray != null ? subArray.filter(item => item !== null).length : 0),
                    0
                );

                // counting the total number of streams that are expected
                const totalStreams = Object.values(streamButtonsRef.current).reduce((total, outerItem) => {
                    return total + outerItem.reduce((innerTotal, innerItem) => {
                        return innerTotal + (Array.isArray(innerItem.mergedIds) ? innerItem.mergedIds.length : 0);
                    }, 0);
                }, 0)

                // if it is too early...
                if (totalReceivedStreams !== totalStreams) {
                    out("[doSave] Tried to save, but it's too early, got data from " + totalReceivedStreams +
                        " streams looking for " + totalStreams);
                    return false;
                }

                out("[doSave] Saving to file...");

                // save
                saveToFile({
                    fileVer: 0,
                    data: ioDataRef.current,
                    agentButtons: agentButtonsRef.current,
                    streamButtons: streamButtonsRef.current,
                    envTitle: envTitleRef.current,
                    envName: envNameRef.current
                }, envTitleRef.current + ".json");

                out("[doSave] Done!");

                // this will signal that we are not interested in saving anymore
                ioDataRef.current = null;

                // restore (remove negative IDs)
                setOpenAgentPanels((prevOpenAgentPanels) => {

                    // this is about the stream buttons for each agent
                    setOpenStreamPanels((prevOpenStreamPanels) => {
                        return Object.keys(prevOpenStreamPanels).reduce((newPanels, agentId) => {
                            const cleanedStreamButtons = prevOpenStreamPanels[agentId].filter(id => id >= 0);
                            if (cleanedStreamButtons.length > 0) {
                                return {...newPanels, [agentId]: cleanedStreamButtons};
                            }
                            return newPanels;
                        }, {});
                    });

                    // this is about agent-related buttons
                    const panels = prevOpenAgentPanels.filter(id => id >= 0);
                    openAgentPanelsRef.current = panels;
                    return panels;
                });

                return true;
            };

            setIsPaused(true);  // this is what actually triggers the re-rendering of the opened figures

            // artificially add negative IDs, to force rendering
            setOpenAgentPanels((prevOpenAgentPanels) => {

                // this is about the stream buttons for each agent
                setOpenStreamPanels((prevOpenStreamPanels) => {
                    return agentButtonsRef.current.reduce((newPanels, button) => {
                        const agentId = button.id; // Invert the sign of button.id
                        const prevOpenStreamPanelsCurAgent =
                            agentId in prevOpenStreamPanels ? prevOpenStreamPanels[agentId] : []

                        const differ = Object.values(streamButtonsRef.current[agentId])
                            .filter(button => !prevOpenStreamPanelsCurAgent.includes(button.id))
                            .map(button => -button.id);

                        // ensure the currentStreamButtons contains all numbers from -1 to -targetLength
                        const updatedStreamButtons =
                            Array.from(new Set([...prevOpenStreamPanelsCurAgent, ...differ]));

                        return {...newPanels, [agentId]: updatedStreamButtons};
                    }, prevOpenStreamPanels);
                });

                // this is about agent-related buttons
                const newIds = agentButtonsRef.current
                    .filter((button) => !prevOpenAgentPanels.includes(button.id)) // only if not already in openAgentPanels
                    .map((button) => -button.id); // invert the sign of each added id

                const panels = [...prevOpenAgentPanels, ...newIds];
                openAgentPanelsRef.current = panels;
                return panels;
            });

            // waiting for the data to be downloaded, stored, and then we can really finish save
            waitingSaveToComplete();
        } else {
            out("[Main] useEffect *** continuing a save procedure *** (skipping, it did not start yet)");
        }
    }, [saving]);

    const startUpload = () => {
        if (!offline) {
            showError("Can only upload in offline mode", "#1F77B4")
        } else {
            out("[Upload] Select file...");
            fileInputRef.current.click();  // this will trigger "handleFileUpload"
        }
    }

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file && file.type === 'application/json') {
            const reader = new FileReader();
            out("[Upload] Loading data...");
            reader.onload = (e) => {
                try {
                    const loadedData = JSON.parse(e.target.result);
                    out("[Upload] Done!");

                    // clear/close all panels
                    setOpenAgentPanels(() => {openAgentPanelsRef.current = []; return [];});
                    setOpenStreamPanels(() => {openStreamPanelsRef.current = {}; return {};});

                    // setting up environment name
                    envNameRef.current = loadedData.envName;
                    setEnvTitle(() => {
                        envTitleRef.current = loadedData.envTitle; return loadedData.envTitle; });

                    // restoring data to be plotted
                    ioDataRef.current = loadedData.data;

                    // agent buttons: restoring icons
                    loadedData.agentButtons = loadedData.agentButtons.map((btn) => ({
                        ...btn,
                        icon: btn.id === 1 ? agentButtonIcons[0] :  // here the ID of the envir is assumed to be 1
                            (btn.authority < 1.0 ? agentButtonIcons[1] : agentButtonIcons[2])
                    }));

                    // agent buttons: setting them up
                    setAgentButtons(() =>{

                        // refreshing reference
                        agentButtonsRef.current = loadedData.agentButtons;
                        return loadedData.agentButtons;
                    });

                    // stream buttons: restoring icons
                    Object.values(loadedData.streamButtons).forEach(streamButtons => {
                        streamButtons.forEach(streamButton => {
                            streamButton.icon = streamButton.mergedLabels[0].endsWith("[y]")
                                ? streamButtonIcons[0]
                                : streamButtonIcons[1];
                        });
                    });

                    // stream buttons: setting them up
                    setStreamButtons(() => {

                        // refreshing reference
                        streamButtonsRef.current = loadedData.streamButtons;
                        return loadedData.streamButtons;
                    });

                    // go
                    setIsPaused(true);
                } catch (error) {
                    console.error("Error while parsing JSON file", error);
                    showError("Error while parsing JSON file");
                }
            };
            reader.readAsText(file);
        } else {
            showError("Please upload a valid JSON file", "#1F77B4");
        }
    };

    const saveOnServer = () => {

        if (!isPaused) {
            showError("Pause the environment first!", "#1F77B4")
            return;
        }
        // meanwhile we fill the stuff to lock the screen
        setSaving(true);

        callAPI('/save', null,
            (x) => {
                if (x !== "<SAVE_OK>") {
                    showError("Unable to save (or to fully save) the environment on server", "#1F77B4")
                }
            },
            () => {
                return true; },
            () => {
                setSaving(false);
            });
    }

    const startSave = () => {
        if (Object.keys(streamButtons).length !== Object.keys(agentButtons).length) {
            showError("Right now there is nothing to save..." +
                "hit the play button at least once to get in touch with the running environment", "#1F77B4")
            return;
        }

        if (!isPaused) {
            showError("Pause the environment first!", "#1F77B4")
            return;
        }

        // before going on, let's appropriately create the ioDataRef "matrix"
        // this will signal that we are interested in saving
        ioDataRef.current =
            new Array(Math.max(0, ...agentButtonsRef.current.map(agentButton => agentButton.id)) + 1)
                .fill(null);

        agentButtonsRef.current.forEach(agentButton => {
            ioDataRef.current[agentButton.id] =
                Array.from({
                        length: Math.max(0,
                            ...(streamButtonsRef.current[agentButton.id].map(streamButton => streamButton.id))) + 1
                    },
                    () => null);
        });

        setIsPaused(false);  // this is to create conditions to trigger the re-rendering of the opened figures

        // meanwhile we fill the stuff to lock the screen
        setSaving(true);
    }

    // downloads the list of streams for a certain agent, and update the list of stream buttons accordingly
    const getStreamsAndUpdateStreamButtons = useCallback((_agentName_, _agentButtonId_) => {

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

        function isEval(label) {
            const match = label.match(/eval(\d+)/);
            if (match) {
                return parseInt(match[1], 10);
            }
            return null;
        }

        function isExpect(label) {
            const match = label.match(/expect(\d+)/);
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
                        !streamButtonsRef.current[_agentButtonId_]
                        || !Array.isArray(streamButtonsRef.current[_agentButtonId_])
                        || streamButtonsRef.current[_agentButtonId_].length === 0
                        || !streamButtonsRef.current[_agentButtonId_].some((streamButton) => (
                            (Array.isArray(streamButton.mergedLabels) && streamButton.mergedLabels.includes(streamName))
                        ))
                );

                // finding the largest ID of the existing stream buttons, also looking inside the mergedIds array
                let maxStreamButtonId = 0;
                if (streamButtonsRef.current[_agentButtonId_]
                    && Array.isArray(streamButtonsRef.current[_agentButtonId_])
                    && streamButtonsRef.current[_agentButtonId_].length > 0) {
                    maxStreamButtonId = streamButtonsRef.current[_agentButtonId_]
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

                    // check stream name: is it a generated/target or eval/expect stream?
                    const generatedNum = isGenerated(newStreamButtons[z].mergedLabels[0]);
                    const targetNum = isTarget(newStreamButtons[z].mergedLabels[0]);
                    const evalNum = isEval(newStreamButtons[z].mergedLabels[0]);
                    const expectNum = isExpect(newStreamButtons[z].mergedLabels[0]);

                    // if the name of the stream is "generatedX" or "targetX", we check if we find the paired stream
                    if ((generatedNum || targetNum) || (evalNum || expectNum)) {

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
                            } else if (targetNum && targetNum >= 0) {

                                // given "generatedX", we want a target that ends with the same "X", and vice-versa
                                if (targetNum !== isGenerated(newStreamButtons[zz].mergedLabels[0])) {
                                    continue;
                                }
                            }

                            // looking for the other stream of the pair
                            if (evalNum && evalNum >= 0) {

                                // given "evalX", we want an "expect" that ends with the same "X", and vice-versa
                                if (evalNum !== isExpect(newStreamButtons[zz].mergedLabels[0])) {
                                    continue;
                                }
                            } else if (expectNum && expectNum >= 0) {

                                // given "expectX", we want an "eval" that ends with the same "X", and vice-versa
                                if (expectNum !== isEval(newStreamButtons[zz].mergedLabels[0])) {
                                    continue;
                                }
                            }

                            const generatedOrEvalZ =
                                ((generatedNum && generatedNum >= 0) || (targetNum && targetNum >= 0)) ?
                                    ((generatedNum && generatedNum >= 0) ? z : zz) : ((evalNum && evalNum >= 0) ? z : zz);
                            const targetOrExpectZ =
                                ((generatedNum && generatedNum >= 0) || (targetNum && targetNum >= 0)) ?
                                    ((generatedNum && generatedNum >= 0) ? zz : z) : ((evalNum && evalNum >= 0) ? zz : z);

                            // if a pair "generatedX" and "targetX" was found... merge!
                            const mergedIds =
                                [...newStreamButtons[generatedOrEvalZ].mergedIds,
                                    ...newStreamButtons[targetOrExpectZ].mergedIds];
                            const mergedLabels =
                                [...newStreamButtons[generatedOrEvalZ].mergedLabels,
                                    ...newStreamButtons[targetOrExpectZ].mergedLabels];

                            // creating the new button about the merged streams
                            const newStreamButton = {
                                id: newStreamButtons[generatedOrEvalZ].id,
                                label: newStreamButtons[generatedOrEvalZ].label,
                                icon: newStreamButtons[generatedOrEvalZ].icon,
                                mergedIds: mergedIds,
                                mergedLabels: mergedLabels,
                                mergedButtons: [],
                                agentButtonId: newStreamButtons[generatedOrEvalZ].agentButtonId
                            };

                            // saving
                            alteredNewStreamButtons.push(newStreamButton)

                            // let's avoid looking again for this button in the original array
                            newStreamButtons[generatedOrEvalZ] = null;
                            newStreamButtons[targetOrExpectZ] = null;
                            isPaired = true;
                            break; // stop searching
                        }
                    }

                    if (!isPaired) {

                        // simple case: nothing to alter, just get the button as it is
                        alteredNewStreamButtons.push(newStreamButtons[z])
                    }
                }

                // updating the current buttons with the newly created ones
                setStreamButtons((prevStreamButtons) => {
                    const updatedStreamButtons = {
                        ...prevStreamButtons,
                        [_agentButtonId_]: [...(prevStreamButtons[_agentButtonId_] || []), ...alteredNewStreamButtons],
                    };

                    // refreshing reference
                    streamButtonsRef.current = updatedStreamButtons;

                    // if we asked to play until a checkpoint, and now we paused, it might be that we reached
                    // such a checkpoint
                    if (serverCommunicatedPlayStepsRef.current === -3) {

                        // if it was a false alarm then playPauseStatus.current.matched_checkpoint_to_show will be null,
                        // otherwise there will be data to show
                        if (playPauseStatusRef.current.matched_checkpoint_to_show !== null) {

                            if (_agentName_ in playPauseStatusRef.current.matched_checkpoint_to_show) {
                                const listOfThingsToShow =
                                    playPauseStatusRef.current.matched_checkpoint_to_show[_agentName_];
                                const numStreamsToShow = listOfThingsToShow
                                    .filter(key => key !== "behavior" && key !== "console").length;

                                let allNewToOpen = [];

                                // parse le list of things to show
                                listOfThingsToShow.forEach(thingToShow => {
                                    if (thingToShow.toLowerCase() === "behavior") {
                                        // skip
                                    } else if (thingToShow.toLowerCase() === "console") {
                                        // skip
                                    } else {

                                        const isSpecialCase =
                                            thingToShow.endsWith(" [y]") || thingToShow.endsWith(" [d]");

                                        // regex to check if stream name ends with " [y]" or " [d]
                                        const regex =
                                            new RegExp(`^${thingToShow.toLowerCase()} \\[[yd]\\]$`);

                                        const streamButtonIDsToShow =
                                            streamButtonsRef.current[_agentButtonId_]
                                                ?.filter((btn) => btn.mergedLabels.some((label) =>
                                                    isSpecialCase ? label.toLowerCase() === thingToShow.toLowerCase()
                                                        : regex.test(label.toLowerCase())))
                                                .map((btn) => btn.id) || [];

                                        const newToOpenFiltered =
                                            streamButtonIDsToShow.filter(item => !allNewToOpen.includes(item));
                                        allNewToOpen = [...allNewToOpen, ...newToOpenFiltered];
                                    }
                                });

                                if (numStreamsToShow > 0 && allNewToOpen.length > 0) {

                                    // open the things that were mentioned in the list
                                    setOpenStreamPanels((prevOpenStreamPanels) => {
                                        return {
                                            ...prevOpenStreamPanels, [_agentButtonId_]:
                                                [...allNewToOpen]
                                        };
                                    });
                                }
                                if (numStreamsToShow === 0) {

                                    // close everything
                                    setOpenStreamPanels((prevOpenStreamPanels) => {
                                        return {
                                            ...prevOpenStreamPanels, [_agentButtonId_]: []
                                        };
                                    });
                                }
                            }
                        }

                        // kill all filters
                        filterStreamButtons(_agentName_, _agentButtonId_,
                            [], showAllRef.current, showAllRef.current);

                    } else {

                        // filter only if not showing a checkpoint
                        filterStreamButtons(_agentName_, _agentButtonId_,
                            shownOwnedStreamsRef.current, shownSignalsRef.current, shownDescriptorsRef.current);
                    }
                    return updatedStreamButtons;
                });
            },
            () => {return !offlineRef.current;},
            () => {
            });
    }, [filterStreamButtons]);  // memoized

    // when paused, we get the list of streams for all the agents of the environment
    useEffect(() => {
        if (!isPaused) {
            out("[Main] useEffect *** fetching data (list of streams for all agents) *** (skipping, not paused)");
            return;
        }

        out("[Main] useEffect *** fetching data (list of streams for all agents) ***");

        agentButtons.forEach((agentButton) => {
            getStreamsAndUpdateStreamButtons(agentButton.label, agentButton.id);
        });
    }, [isPaused, agentButtons, getStreamsAndUpdateStreamButtons]);  // last one is a memoized function

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
                        setIsPaused(() => {
                            setPlayPauseStatus(() => {
                                playPauseStatusRef.current = x;
                                return x;
                            });
                            return false;
                        });
                    } else if (x.status === "paused") {

                        // if we asked to play until a checkpoint, and now we paused, it might be that we reached
                        // such a checkpoint: let's open those agent panels that are involved in the checkpoint,
                        // and let's close the ones that are not involved (do the same for consoles and FSMs)
                        if (serverCommunicatedPlayStepsRef.current === -3) {

                            // if it was a false alarm, then x.matched_checkpoint_to_show will be null,
                            // otherwise there will be data to show
                            if (x.matched_checkpoint_to_show !== null) {

                                // opening the agent-panels that are mentioned in the checkpoint
                                // and closing the other ones
                                const agentButtonIDsToShow =
                                    Object.keys(x.matched_checkpoint_to_show).map((key) =>
                                        agentButtonsRef.current.find((btn) => btn.label === key)?.id)
                                        .filter(Boolean);
                                const newToOpen = agentButtonIDsToShow.filter(item =>
                                    !openAgentPanelsRef.current.includes(item));
                                const existingToKeep = openAgentPanelsRef.current.filter(item =>
                                    agentButtonIDsToShow.includes(item));
                                if (newToOpen.length > 0 ||
                                    (existingToKeep.length > 0 &&
                                        existingToKeep.length < openAgentPanelsRef.current.length)) {
                                    setOpenAgentPanels(() => {
                                        const panels = [...existingToKeep, ...newToOpen];
                                        openAgentPanelsRef.current = panels;
                                        return panels;
                                    });
                                }

                                // behaviors and consoles
                                Object.entries(x.matched_checkpoint_to_show).forEach(([agentName,
                                                                                          listOfThingsToShow]) => {
                                    let closeBehavior = true;
                                    let closeConsole = true;

                                    // find agent button ID
                                    const agentButtonId =
                                        agentButtonsRef.current.find((btn) => btn.label === agentName)?.id;

                                    // parse le list of things to show
                                    listOfThingsToShow.forEach(thingToShow => {

                                        if (thingToShow.toLowerCase() === "behavior") {
                                            closeBehavior = false;
                                            if (!openFSMPanelsRef.current.includes(agentButtonId)) {
                                                setOpenFSMPanels((prev) => {
                                                    const panels = [...prev, agentButtonId];
                                                    openFSMPanelsRef.current = panels;
                                                    return panels;
                                                });
                                            }

                                        } else if (thingToShow.toLowerCase() === "console") {
                                            closeConsole = false;
                                            if (!openConsolePanelsRef.current.includes(agentButtonId)) {
                                                setOpenConsolePanels((prev) => {
                                                    const panels = [...prev, agentButtonId];
                                                    openConsolePanelsRef.current = panels;
                                                    return panels;
                                                });
                                            }
                                        }
                                    });

                                    // close the things that were not mentioned in the list
                                    if (closeBehavior) {
                                        setOpenFSMPanels((prev) => {
                                            const panels = prev.filter((pid) => pid !== agentButtonId);
                                            openFSMPanelsRef.current = panels;
                                            return panels;
                                        });
                                    }

                                    if (closeConsole) {
                                        setOpenConsolePanels((prev) => {
                                            const panels = prev.filter((pid) => pid !== agentButtonId);
                                            openConsolePanelsRef.current = panels;
                                            return panels;
                                        });
                                    }
                                });
                            }
                        }

                        if (x.more_checkpoints_available && !firstCheckpointWasAlreadyFoundRef.current) {
                            setSelectedPlayOption("\u2714");
                            firstCheckpointWasAlreadyFoundRef.current = true;
                        }

                        setIsPaused(() => {
                            setPlayPauseStatus(() => {
                                playPauseStatusRef.current = x;
                                return x;
                            });
                            return true;
                        });
                    } else if (x.status === "ended") {
                        setIsPaused(() => {
                            setPlayPauseStatus(() => {
                                playPauseStatusRef.current = x;
                                return x;
                            });
                            return true;
                        });
                    } else {
                        throw new Error("Unknown status: " + x.status);
                    }
                },
                () => {
                    if (_errorCallback_) {
                        _errorCallback_();
                    }
                    return !offlineRef.current;
                },
                () => {
                    playPauseStatusAskedRef.current = false;
                });
        }
    }

    // handle the click on the play/pause button
    const handleClickOnPlayPauseButton = () => {

        // if something is drawing/working/fetching-data, do not let it go
        if (isBusy > 0) {
            out("[Main] *** click on play/pause button *** (ignored due to other components busy)");
            return;
        } else {
            out("[Main] *** click on play/pause button ***");
        }

        if (playPauseStatus.status === "paused") {

            // getting play options (number of steps to run)
            serverCommunicatedPlayStepsRef.current = selectedPlayOption.endsWith("k") ?
                parseInt(selectedPlayOption.replace('k', '')) * 1000 :
                selectedPlayOption === "1S" ? -1 : selectedPlayOption === "\u221E" ? -2 :
                    selectedPlayOption === "\u2714" ? -3 : parseInt(selectedPlayOption)

            // asking to play
            out("[Main] *** fetching data (ask-to-play for " +
                serverCommunicatedPlayStepsRef.current + " steps) ***");
            callAPI('/ask_to_play', "steps=" + serverCommunicatedPlayStepsRef.current,
                (x) => {
                    out("[Main] server responded to the ask-to-play request, " +
                        "which was received by the sever when at step id: " + x);
                    startOrStopPlayTimer();  // starting timer!
                },
                () => {
                    return !offlineRef.current;
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
                    return !offlineRef.current;
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
            {saving && <div className="saving-spinner"></div>}

            <div className="p-6 space-y-2 flex flex-col items-center w-full">
                <div className="flex flex-col items-center justify-center text-center">
                    <h1 className="text-2xl font-semibold mt-2 text-center">
                        NARNIAN: {envTitle}
                        <button onClick={saveOnServer}
                                className={`save-button inline-flex items-center pl-2 gap-2 relative top-0.5 
                                ${offline ? "hidden" : ""}`}><Save size={20}/></button>
                        <button onClick={!offline ? startSave : startUpload}
                                className="save-button inline-flex items-center pl-2 gap-2 relative top-0.5">
                            {!offline ? <Download size={20}/> : <Upload size={20}/>}</button>
                        <input type="file" accept=".json" ref={fileInputRef} style={{display: "none"}}
                               onChange={handleFileUpload}
                        />
                    </h1>
                    <span className={`text-sm mt-1 ${!playPauseStatus ? "hidden" : ""}`}>
                            Environment Time: {playPauseStatus.time.toFixed(2)}s
                    </span>
                </div>

                <div className="flex flex-wrap gap-4 w-full mt-0 pt-0 justify-center">

                    <div className="flex items-center"
                         style={{display: playPauseStatus.status === 'ended' ? 'none' : 'flex' }}>

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
                                                    ) :  "bg-blue-400" // "unexpected status!"
                                            )
                                    )
                            }`}
                        >
                        </button>

                    </div>

                    <button onClick={handleClickOnPlayPauseButton}
                            className={`px-4 py-2 rounded-2xl bg-amber-200 
                            ${isBusy > 0 ? 
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
                        {(playPauseStatus.more_checkpoints_available ?
                            ["\u2714", "1S", "1", "100", "1k", "100k", "\u221E"] :
                        ["1S", "1", "100", "1k", "100k", "\u221E"]).map((option) => (
                            <button key={option} onClick={() => setSelectedPlayOption(option)}
                                className={selectedPlayOption === option ?
                                    "h-6 text-sm bg-amber-200 hover:bg-amber-300 " +
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

                <div className={`grid ${gridCols} gap-8 w-full pt-4`}>
                    {agentButtons.map((agent_button) => {
                        const isOpen = openAgentPanels.includes(agent_button.id);
                        const isForcedOpen = openAgentPanels.includes(-agent_button.id);
                        const shouldRender = isOpen || isForcedOpen;
                        const shouldHideOthers = isForcedOpen && !isOpen;

                        return shouldRender && (  // **** big thing opening here.... ***
                            <AnimatePresence key={agent_button.id}>
                                <motion.div initial={{opacity: 0, height: 0}}
                                            animate={{opacity: 1, height: shouldHideOthers ? 0 : "auto"}}
                                            exit={{opacity: 0, height: 0}}
                                            transition={{duration: 0.3}}
                                            className={`${bgColors[(agent_button.id - 1) % bgColors.length]} w-full 
                                            p-4 rounded-2xl shadow-lg border 
                                                ${shouldHideOthers ? "hidden" : ""}`}>

                                    <h2 className="font-medium text-lg flex items-center justify-between">
                                        <div className="flex items-center">
                                            <span className="mr-1 ml-5 text-xl">
                                                {agent_button.label}
                                            </span>
                                            <Balloon _agentName_={agent_button.label}
                                                     _isPaused_={isPaused}
                                                     _isBusyRef_={isBusyRef}
                                                     _setIsBusy_={setIsBusy}
                                            />
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <button
                                                className={`w-6 h-6 pb-0
                                                ${openFSMPanels.includes(agent_button.id) ?
                                                    "text-white bg-blue-500" : "bg-gray-100"} rounded-full 
                                                    flex items-center justify-center ml-2
                                                    ${offline ? "hidden" : ""}`}
                                                onClick={() => toggleFSMPanel(agent_button.id)}>
                                                B
                                            </button>
                                            <button
                                                className={`w-6 h-6 pb-0
                                                ${openConsolePanels.includes(agent_button.id) ?
                                                    "text-white bg-blue-500" : "bg-gray-100"} rounded-full flex 
                                                    items-center justify-center ml-2
                                                    ${offline ? "hidden" : ""}`}
                                                onClick={() => toggleConsolePanel(agent_button.id)}>
                                                C
                                            </button>
                                            <button
                                                className={`w-6 h-6 top font-medium flex rounded-full text-white 
                                            ${shownSignals.includes(agent_button.id) ?
                                                    "text-white bg-blue-500" : "text-black bg-gray-100"}
                                             items-center justify-center ml-2 pb-0 ${offline ? "hidden" : ""}`}
                                                onClick={() =>
                                                    toggleShowSignals(agent_button.label, agent_button.id)}>
                                                <Activity size={16}
                                                          className={`${shownSignals.includes(agent_button.id) ? 
                                                          "text-white" : "text-black"}`} />
                                            </button>
                                            <button
                                                className={`w-6 h-6 top font-medium flex rounded-full text-white 
                                            ${shownDescriptors.includes(agent_button.id) ?
                                                    "text-white bg-blue-500" : "text-black bg-gray-100"}
                                             items-center justify-center ml-2 pb-0 ${offline ? "hidden" : ""}`}
                                                onClick={() =>
                                                    toggleShowDescriptors(agent_button.label, agent_button.id)}>
                                                <Search size={16}
                                                        className={`${shownDescriptors.includes(agent_button.id) ? 
                                                          "text-white" : "text-black"}`} />
                                            </button>
                                            <button
                                                className={`w-6 h-6 top font-medium flex rounded-full text-white 
                                            ${shownOwnedStreams.includes(agent_button.id) ?
                                                    "text-white bg-blue-500" : "text-black bg-gray-100"}
                                             items-center justify-center ml-2 pb-0 ${offline ? "hidden" : ""}`}
                                                onClick={() =>
                                                    toggleShowOwnedStreams(agent_button.label, agent_button.id)}>
                                                <Hand size={16}
                                                      className={`${shownOwnedStreams.includes(agent_button.id) ? 
                                                          "text-white" : "text-black"}`} />
                                            </button>
                                        </div>
                                    </h2>

                                    <StreamButtonContainer
                                        _streamButtons_={streamButtons}
                                        _agentButton_={agent_button}
                                        _visibleStreamButtons_={visibleStreamButtons}
                                        _handleClick_={handleClick}
                                        _handleDoubleClick_={handleDoubleClick}
                                        _handleDrop_={handleDrop}
                                        _checkIfActive_={checkIfActive}/>

                                    <div className={`gap-4 mt-6  
                                        ${(() => {
                                        const numOpened =
                                            openFSMPanels.includes(agent_button.id) +
                                            openConsolePanels.includes(agent_button.id) +
                                            (openStreamPanels[agent_button.id]?.filter((id) => id > 0).length
                                                || 0);
                                        return `grid ${numOpened === 1 ? "sm:grid-cols-1 max-w-[900px] mx-auto" :
                                            ((numOpened === 2 ||
                                                openAgentPanels?.filter((id) => id > 0).length > 1) ?
                                                "sm:grid-cols-2" :
                                                "sm:grid-cols-3")}`;
                                    })()}`}>
                                        {openFSMPanels.includes(agent_button.id) &&
                                            <div className="h-[500px] w-full flex justify-center">
                                                <div className="w-full p-0 pt-4 pb-5 bg-gray-100
                                                        rounded-xl shadow text-center">
                                                    <h3 className="font-medium">Behavior</h3>
                                                    <FSM _agentName_={agent_button.label}
                                                         _isPaused_={isPaused}
                                                         _isBusyRef_={isBusyRef}
                                                         _setIsBusy_={setIsBusy}
                                                    />
                                                </div>
                                            </div>
                                        }
                                        {openConsolePanels.includes(agent_button.id) &&
                                            <div className="h-[500px] w-full flex justify-center">
                                                <div className="w-full p-0 pt-4 pb-5 bg-gray-100
                                                        rounded-xl shadow text-center">
                                                    <h3 className="font-medium">Console</h3>
                                                    <Console _agentName_={agent_button.label}
                                                             _isPaused_={isPaused}
                                                             _isBusyRef_={isBusyRef}
                                                             _setIsBusy_={setIsBusy}
                                                    />
                                                </div>
                                            </div>
                                        }
                                        {openStreamPanels[agent_button.id]?.map((id) => {
                                            const shouldHidePlotFigure = id < 0;
                                            if (id < 0) {
                                                id = -id;
                                            }

                                            const streamButton = streamButtons[agent_button.id]?.find(
                                                (btn) => btn.id === id
                                            );
                                            return (
                                                <div key={id}
                                                     style={{visibility: shouldHidePlotFigure ? "hidden" : "visible"}}
                                                     className={`min-h-[500px] p-0 pt-4 pb-8 bg-gray-50 border
                                                    rounded-xl shadow text-center 
                                                    ${shouldHidePlotFigure ? "hidden" : ""}`}>
                                                    <h3 className="font-medium flex items-center justify-center">
                                                        <span className="w-5 h-5">{streamButton?.icon}</span>
                                                        <span className="ml-1">{streamButton?.label}</span>
                                                    </h3>
                                                    <PlotFigure _agentName_={agent_button.label}
                                                                _streamStruct_={streamButton}
                                                                _isPaused_={isPaused}
                                                                _isBusyRef_={isBusyRef}
                                                                _setIsBusy_={setIsBusy}
                                                                _ioDataRef_={ioDataRef}
                                                                _offline_={offlineRef.current}
                                                                _yMin_={playPauseStatus.y_range[0]}
                                                                _yMax_={playPauseStatus.y_range[1]}
                                                    />
                                                </div>
                                            );
                                        })}
                                    </div>
                                </motion.div>
                            </AnimatePresence>
                        );  // **** big thing closing here.... ***
                    })}
                </div>
            </div>
            <div className="flex items-center justify-center w-full"><span className="text-sm">
                <a href="https://cai.diism.unisi.it/" className="text-blue-800 hover:underline">
                    Collectionless AI Team</a> - &#169; Stefano Melacci (2025)</span></div>
        </DndProvider>
    );
}

// create a new stream button with several action handlers attached (given a streamButton structure)
// it is interpreted and built as a React component: keep name with the starting capital letter (otherwise it fails!)
function DraggableStreamButton({_streamButton_, _onDrop_, _onDoubleClick_, _onClick_, _checkIfActive_, _visible_}) {

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
            className={`${_visible_ ? "" : "hidden"} flex h-6 items-center justify-center px-3 py-4 rounded-2xl 
            shadow-md select-none cursor-move 
            text-center transition-colors whitespace-nowrap ${
                _checkIfActive_() ? "bg-blue-600 text-white" : "bg-gray-50 hover:bg-gray-200"
            } ${isDragging ? "opacity-50" : "opacity-100"}`}
            onClick={() => _onClick_(_streamButton_)}
            onDoubleClick={() => _onDoubleClick_(_streamButton_)}
            ref={(node) => {
                if (node) drag(drop(node));
            }}
        >
            <span className="w-5 h-5 flex items-center justify-center">{_streamButton_.icon}</span>
            <span className="ml-1">{_streamButton_.label}</span>
        </div>
    );
}

const StreamButtonContainer = ({ _streamButtons_, _agentButton_, _visibleStreamButtons_,
                                   _handleClick_, _handleDoubleClick_, _handleDrop_, _checkIfActive_ }) => {
    const containerRef = useRef(null);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(false);

    out("[StreamButtonContainer] _visibleStreamButtons_=" + JSON.stringify(_visibleStreamButtons_));

    // check if scrolling is possible
    const updateScrollState = () => {
        if (containerRef.current) {
            const { scrollLeft, scrollWidth, clientWidth } = containerRef.current;
            setCanScrollLeft(scrollLeft > 0);
            setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 1);
        }
    };

    // scroll left or right
    const scroll = (direction) => {
        if (containerRef.current) {
            const scrollAmount = containerRef.current.clientWidth / 2; // Scroll by half the container width
            containerRef.current.scrollBy({ left: direction * scrollAmount, behavior: "smooth" });
        }
    };

    // update scroll state on mount and on resize
    useEffect(() => {
        updateScrollState();
        const { scrollLeft, scrollWidth, clientWidth } = containerRef.current;
        out("[StreamButtonContainer] useEffect *** updating scroll state to scrollLeft=" +
            scrollLeft + ", scrollWidth=" + scrollWidth + ", clientWidth=" + clientWidth + " ***");
        window.addEventListener("resize", updateScrollState);
        return () => window.removeEventListener("resize", updateScrollState);
    }, [_visibleStreamButtons_]);

    return (
        <div className="relative flex items-center w-full pt-7 pb-0">
            {_visibleStreamButtons_[_agentButton_.id]?.length > 0 ? "Streams:" : ""}
            {canScrollLeft && (
                <button className="border border-black absolute top left-[68px] z-10 bg-white shadow-md rounded-full p-2"
                        onClick={() => scroll(-1)}>
                    <ChevronLeft size={24}/>
                </button>
            )}

            <div ref={containerRef}
                 className="flex gap-4 justify-start w-full overflow-x-auto scrollbar-hide
                 scroll-smooth whitespace-nowrap px-10 pt-2 pb-2 mr-2 ml-6 mt-0" onScroll={updateScrollState}>
                {_streamButtons_[_agentButton_.id]?.map((streamButton) => (
                    <AnimatePresence key={streamButton.mergedIds.join("-")}>
                        <motion.div
                            initial={{opacity: 0, scale: 0.9}}
                            animate={{opacity: 1, scale: 1}}
                            exit={{opacity: 0, scale: 0.9}}
                            transition={{duration: 0.2}}
                        >
                            <DraggableStreamButton
                                _streamButton_={streamButton}
                                _onDrop_={(droppedStreamButton) =>
                                    _handleDrop_(
                                        droppedStreamButton.agentButtonId,
                                        droppedStreamButton.id,
                                        streamButton.agentButtonId,
                                        streamButton.id
                                    )
                                }
                                _onDoubleClick_={() =>
                                    _handleDoubleClick_(streamButton.agentButtonId, streamButton.id)
                                }
                                _onClick_={() =>
                                    _handleClick_(streamButton.agentButtonId, streamButton.id)
                                }
                                _checkIfActive_={() =>
                                    _checkIfActive_(streamButton.agentButtonId, streamButton.id)
                                }
                                _visible_={_visibleStreamButtons_[streamButton.agentButtonId]
                                    .includes(streamButton.id)}
                            />
                        </motion.div>
                    </AnimatePresence>
                ))}
            </div>

            {canScrollRight && (
                <button className="border border-black absolute top right-0 z-10 bg-white shadow-md rounded-full p-2"
                        onClick={() => scroll(1)}>
                    <ChevronRight size={24}/>
                </button>
            )}
        </div>
    );
};
