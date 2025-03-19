import {useState, useEffect, useRef} from "react";
import {motion} from "framer-motion";
import {callAPI, out} from "./utils";

export default function Balloon({_agentName_, _isPaused_, _setBusy_}) {
    out("[Balloon] " +
        "_agentName_: " + _agentName_ + ", " +
        "_isPaused_: " + _isPaused_);

    // text on the balloon
    const [displayedText, setDisplayedText] = useState("");

    // text from the net
    const [receivedText, setReceivedText] = useState("");
    const receivedTextRef = useRef(receivedText);

    const irregularVerbsRef = useRef({
        "arising": "arisen",
        "awakening": "awoken",
        "being": "been",
        "bearing": "borne",
        "becoming": "become",
        "beginning": "begun",
        "bending": "bent",
        "betting": "bet",
        "binding": "bound",
        "biting": "bitten",
        "bleeding": "bled",
        "blowing": "blown",
        "breaking": "broken",
        "bringing": "brought",
        "building": "built",
        "burning": "burnt",
        "bursting": "burst",
        "buying": "bought",
        "catching": "caught",
        "choosing": "chosen",
        "coming": "come",
        "costing": "cost",
        "cutting": "cut",
        "dealing": "dealt",
        "digging": "dug",
        "doing": "done",
        "drawing": "drawn",
        "drinking": "drunk",
        "driving": "driven",
        "eating": "eaten",
        "falling": "fallen",
        "feeding": "fed",
        "feeling": "felt",
        "fighting": "fought",
        "finding": "found",
        "fleeing": "fled",
        "flinging": "flung",
        "flying": "flown",
        "forgetting": "forgotten",
        "forgiving": "forgiven",
        "freezing": "frozen",
        "getting": "got",
        "giving": "given",
        "going": "gone",
        "growing": "grown",
        "hanging": "hung",
        "having": "had",
        "hearing": "heard",
        "hiding": "hidden",
        "holding": "held",
        "keeping": "kept",
        "knowing": "known",
        "laying": "laid",
        "leading": "led",
        "leaping": "leapt",
        "learning": "learnt",
        "leaving": "left",
        "lending": "lent",
        "letting": "let",
        "lying": "lain",
        "losing": "lost",
        "making": "made",
        "meaning": "meant",
        "meeting": "met",
        "paying": "paid",
        "proving": "proven",
        "putting": "put",
        "reading": "read",
        "riding": "ridden",
        "rising": "risen",
        "running": "run",
        "saying": "said",
        "seeing": "seen",
        "seeking": "sought",
        "selling": "sold",
        "sending": "sent",
        "setting": "set",
        "shaking": "shaken",
        "shedding": "shed",
        "shining": "shone",
        "shooting": "shot",
        "showing": "shown",
        "shutting": "shut",
        "singing": "sung",
        "sinking": "sunk",
        "sitting": "sat",
        "sleeping": "slept",
        "sliding": "slid",
        "speaking": "spoken",
        "spending": "spent",
        "spinning": "spun",
        "spitting": "spat",
        "splitting": "split",
        "spoiling": "spoilt",
        "standing": "stood",
        "stealing": "stolen",
        "sticking": "stuck",
        "striking": "struck",
        "striving": "striven",
        "swearing": "sworn",
        "sweeping": "swept",
        "swimming": "swum",
        "swinging": "swung",
        "taking": "taken",
        "teaching": "taught",
        "tearing": "torn",
        "telling": "told",
        "thinking": "thought",
        "throwing": "thrown",
        "understanding": "understood",
        "upsetting": "upset",
        "waking": "woken",
        "wearing": "worn",
        "weaving": "woven",
        "weeping": "wept",
        "winning": "won",
        "wringing": "wrung",
        "writing": "written"
    });

    useEffect(() => {
        out("[Balloon] useEffect *** printing ***");

        setDisplayedText(""); // reset text when input changes
        if (!receivedText || receivedText.length === 0)
            return;

        let index = 0;
        const interval = setInterval(() => {
            setDisplayedText((prev) => receivedText.slice(0, index + 1));
            index++;
            if (index >= receivedText.length) {
                clearInterval(interval);
            }
            else if (index >= 250) {
                setDisplayedText((prev) => receivedText.slice(0, index + 1) + "...");
                clearInterval(interval);
            }
        }, 100); // 0.1 seconds per character

        return () => clearInterval(interval);
    }, [receivedText]);

    useEffect(() => {
        if (!_isPaused_) {
            out("[Balloon] useEffect *** fetching data (get last console message) *** (skipping, not paused)");
            return;
        }

        // this will tell the parent that this component is working
        _setBusy_((prev) => prev + 1);

        out("[Balloon] useEffect *** fetching data (get last console message, agent_name: " + _agentName_ + ") ***");
        callAPI('/get_console', {agent_name: _agentName_, last_only: true},
            (x) => {
                // the expected format of the received data, that is a circular buffer that does not start necessarily
                // from zero, is:
                // {"output_messages_last_pos": x, "output_messages_count": y, "output_messages", [...]}

                function capitalize(word) {
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }

                function ing2ed(word) {
                    return word.slice(0, -3) + "ed";
                }

                // if starts with <FAILED>, the previously received text is kept
                const text =
                    x.output_messages && x.output_messages[0]
                    && !x.output_messages[0].startsWith("<FAILED>") ? x.output_messages[0] : receivedTextRef.current;
                const isInAction = x.behav_status.action !== null;
                const stateWithAction = x.behav_status.state_with_action;

                if (text && text.length > 0) {

                    const words = text.split(" ");
                    if (words.length === 0) return;

                    let firstWord = words[0].toLowerCase();

                    if (!isInAction && !stateWithAction && firstWord.endsWith("ing")) {
                        if (irregularVerbsRef.current[firstWord]) {
                            words[0] = capitalize(irregularVerbsRef.current[firstWord]);
                        } else {
                            words[0] = capitalize(ing2ed(firstWord));
                        }
                    } else {
                        words[0] = capitalize(firstWord);
                    }

                    setReceivedText(() => {
                        const newText = words.join(" ");
                        receivedTextRef.current = newText;
                        return newText;
                    });
                }
            },
            () => {
                setReceivedText(() => {
                    const newText = "Error while getting text";
                    receivedTextRef.current = newText;
                    return newText;
                });
                return true;
            },
            () => {
                // this will tell the parent that this component is now ready
                _setBusy_((prev) => prev - 1);
            }
        );
    }, [_isPaused_, _agentName_, _setBusy_]);
    // listen to the pause state (_isPaused_), while _agentName_ and _setBusy_ are not going to change

    return (
        <div className="relative">
            <div className="absolute top-1/2 transform left-7 -translate-y-1/2 z-50 max-h-48">
                <div
                    className={`relative flex justify-center items-center w-[400px] min-h-16 max-h-48 bg-white border-2 border-black
            rounded-full p-4 shadow-lg ${(!receivedText || receivedText.length === 0) ? "hidden": ""}`}>
                    <motion.p
                        className="text-center text-base"
                        initial={{opacity: 0}}
                        animate={{opacity: 1}}
                    >
                        {displayedText}
                    </motion.p>

                    {/* arrow pointing left */}
                    <div className="absolute right-full top-1/2 transform -translate-y-1/2 w-0 h-0
                    border-r-[20px] border-r-black
                    border-t-[10px] border-t-transparent
                    border-b-[10px] border-b-transparent">
                    </div>
                </div>
            </div>
        </div>
    );
}
