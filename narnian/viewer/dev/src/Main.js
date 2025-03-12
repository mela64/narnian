import {useState, useRef, useCallback, useEffect} from "react";  // react
import {DndProvider, useDrag, useDrop} from "react-dnd";  // still about react
import {HTML5Backend} from "react-dnd-html5-backend";  // still about react
import {motion, AnimatePresence} from "framer-motion";  // animated opening panels
import FSM from "./FSM";
import Console from "./Console";
import PlotFigure from "./PlotFigure";
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
    Download
} from "lucide-react";

let clickTimeout; // timer that triggers a reaction to the click action
let clickCount = 0;  // number of consecutive clicks (to distinguish clicks from double clicks)

// icons used in the agent buttons or stream buttons
const agentButtonIcons = [<Settings/>, <Bot/>, <GraduationCap/>];
const streamButtonIcons = [<Activity/>, <Search/>, <Waves/>];

// the structure representing the play/pause status, in its initial (unknown) setting
const unknownPlayPauseStatus = {
    "status": "?",
    "still_to_play": -1
}

// the structure representing the pause status
const endedStatus = {
    "status": "ended",
    "still_to_play": -1
}

export default function Main() {

    // working-state of the components used in this page
    const [isFSMBusy, setIsFSMBusy] = useState(0);
    const [isConsoleBusy, setIsConsoleBusy] = useState(0);
    const [isPlotFigureBusy, setIsPlotFigureBusy] = useState(0);

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

    // the name of the environment, set to "?" when unknown
    const [envTitle, setEnvTitle] = useState("?");
    const envTitleRef = useRef(envTitle);
    const envNameRef = useRef("?");

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

    // up-lifted data from the plot figures (array of arrays of integers)
    const ioDataRef = useRef(null)

    // saving flag
    const [saving, setSaving] = useState(false);

    // if offline
    const [offline, setOffline] = useState(false);
    const offlineRef = useRef(offline);
    const fileInputRef = useRef(null);

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

    // keeping references up-to-date at each rendering operation
    useEffect(() => {
        out("[Main] useEffect *** updating envTitleRef ***");
        envTitleRef.current = envTitle;
    }, [envTitle]);

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
            (x) => { envNameRef.current = x.name; setEnvTitle(x.title); },
            () => { envNameRef.current = "?"; setEnvTitle("Offline");
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

                    // if it is the first time (only!), we collect the button IDs in the other panel-associated lists
                    if (firstTime) {
                        setOpenFSMPanels(agent_buttons.map(agent_button => agent_button.id));
                        setOpenConsolePanels(agent_buttons.map(agent_button => agent_button.id));
                    }

                    setAgentButtons(agent_buttons);
                }
            },
            () => {
                if (!offlineRef.current) {
                    setAgentButtons([]); return true;
                } else { return false; }},
            () => {});
    }, [envTitle, isPaused]);  // when the status of the loading env name operation changes, we get the agent list
    // it the authority changed, we need to update button graphics, so we also call this API when isPaused

    // when paused, we get the list of streams for all the agents of the environment
    useEffect(() => {
        if (!isPaused) {
            out("[Main] useEffect *** fetching data (list of streams for all agents) *** (skipping, not paused)");
            return;
        }

        out("[Main] useEffect *** fetching data (list of streams for all agents) ***");

        agentButtons.forEach((agentButton) => {
            getStreamsAndUpdateStreamButtons(agentButton.label, agentButton.id, streamButtonsRef.current);
        });
    }, [isPaused, agentButtons]);

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
                    setPlayPauseStatus(unknownPlayPauseStatus);
                } else {
                    setIsPaused(() => { setPlayPauseStatus(endedStatus); return true; } );
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
        setStreamButtons((prev) => ({
            ...prev,
            [_agentButtonIdOfClicked_]: [
                ...prev[_agentButtonIdOfClicked_].filter((btn) => btn.id !== streamButtonClicked.id),
                ...restoredButtons, // Add the individual buttons to the specific panel
            ],
        }));
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
                    return prevOpenAgentPanels.filter(id => id >= 0);
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

                return [...prevOpenAgentPanels, ...newIds];
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
                    setOpenAgentPanels([]);
                    setOpenStreamPanels({});

                    // setting up environment name
                    envNameRef.current = loadedData.envName;
                    setEnvTitle(loadedData.envTitle);

                    // restoring data to be plotted
                    ioDataRef.current = loadedData.data;

                    // agent buttons: restoring icons
                    loadedData.agentButtons = loadedData.agentButtons.map((btn, index) => ({
                        ...btn,
                        icon: btn.id === 1 ? agentButtonIcons[0] :  // here the ID of the envir is assumed to be 1
                            (btn.authority < 1.0 ? agentButtonIcons[1] : agentButtonIcons[2])
                    }));

                    // agent buttons: setting them up
                    setAgentButtons(loadedData.agentButtons);

                    // stream buttons: restoring icons
                    Object.values(loadedData.streamButtons).forEach(streamButtons => {
                        streamButtons.forEach(streamButton => {
                            streamButton.icon = streamButton.mergedLabels[0].endsWith("[y]")
                                ? streamButtonIcons[0]
                                : streamButtonIcons[1];
                        });
                    });

                    // stream buttons: setting them up
                    setStreamButtons(loadedData.streamButtons);

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
            () => {
                if (!offlineRef.current) { setStreamButtons((prev) => (prev)); return true; }
                else { return false; }},
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
                        setIsPaused(() => { setPlayPauseStatus(x); return false; } );
                    } else if (x.status === "paused") {
                        setIsPaused(() => { setPlayPauseStatus(x); return true; } );
                    } else if (x.status === "ended") {
                        setIsPaused(() => { setPlayPauseStatus(x); return true; } );
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
        if (isFSMBusy > 0 || isConsoleBusy > 0 || isPlotFigureBusy > 0) {
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

            <div className="p-6 space-y-8 flex flex-col items-center w-full">
                <div className="flex flex-col items-center justify-center text-center">
                    <h1 className="text-2xl font-semibold mt-2">NARNIAN</h1>
                    <h1 className="text-2xl font-semibold mt-2">Environment:{" "}
                        {envTitle}
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
                </div>

                <div className="flex flex-wrap gap-4 w-full justify-center">

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
                                                    ) :  "bg-gray-400" // "unexpected status!"
                                            )
                                    )
                            }`}
                        >
                        </button>

                    </div>

                    <button onClick={handleClickOnPlayPauseButton}
                            className={`px-4 py-2 rounded-2xl bg-amber-200 
                            ${(isFSMBusy > 0 || isConsoleBusy > 0 || isPlotFigureBusy > 0) ? 
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
                                            className={`w-full p-4 bg-white rounded-2xl shadow-lg border space-y-6 
                                                ${shouldHideOthers ? "hidden" : ""}`}>

                                    <h2 className="font-medium text-lg flex items-center justify-center">
                                        <span className="mr-1">{agent_button.label}</span>
                                        <button
                                            className={`w-6 h-6 
                                                ${openFSMPanels.includes(agent_button.id) ?
                                                "text-white bg-blue-500" : "bg-gray-100"} rounded-full 
                                                    flex items-center justify-center ml-2
                                                    ${offline ? "hidden" : ""}`}
                                            onClick={() => toggleFSMPanel(agent_button.id)}>
                                            B
                                        </button>
                                        <button
                                            className={`w-6 h-6 
                                                ${openConsolePanels.includes(agent_button.id) ?
                                                "text-white bg-blue-500" : "bg-gray-100"} rounded-full flex 
                                                    items-center justify-center ml-2
                                                    ${offline ? "hidden" : ""}`}
                                            onClick={() => toggleConsolePanel(agent_button.id)}>
                                            C
                                        </button>
                                    </h2>

                                    <div className={`grid grid-cols-1 ${(openFSMPanels.includes(agent_button.id) &&
                                        openConsolePanels.includes(agent_button.id)) ?
                                        "sm:grid-cols-2" : "sm:grid-cols-1"} gap-4 ${offline ? "hidden" : ""}`}>

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
                                        ${openStreamPanels[agent_button.id]?.filter((id) => id > 0)
                                        .length <= 2 ?
                                        (openStreamPanels[agent_button.id]?.filter((id) => id > 0)
                                            .length <= 1 ?
                                            " grid sm:grid-cols-1 max-w-[900px] mx-auto"
                                            : "grid sm:grid-cols-2") : "grid sm:grid-cols-3"}`}>
                                        {openStreamPanels[agent_button.id]?.map((id) => {
                                            const shouldHidePlotFigure = id < 0;
                                            if (id < 0) { id = -id; }

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
                                                                _setBusy_={setIsPlotFigureBusy}
                                                                _ioDataRef_={ioDataRef}
                                                                _offline_={offlineRef.current}
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
