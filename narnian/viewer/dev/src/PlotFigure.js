import {useRef, useState, useEffect} from "react";
import Plot from "react-plotly.js";
import {callAPI, out} from "./utils";

const maxTotalPlots = 8; // customizable (total number of plots in the same figure)
const maxVecComponentsPerSignal = 6; // customizable (max number of components from a vectorial signal)

export default function PlotFigure({ _agentName_, _streamName_, _isPaused_, _setBusy_ }) {
    out("[PlotFigure] " +
        "_agentName_: " + _agentName_ + ", " +
        "_streamName_: " + _streamName_ + ", " +
        "_isPaused_: " + _isPaused_);

    // basic structure of the data that will be plotted: it is an array where each element is the data of a single plot
    // in particular, each element is {x: [...], y: [...], type: "scatter", mode: "lines", name: "plot-name"}
    const emptyPlotData = Array.from({
        length: 0
    }, (_,) => ({
        x: [],
        y: [],
        type: "scatter",
        mode: "lines",
        name: null,
    }));

    // the name of the stream of this figure (it can be a plain name, such as "sin", or the name of a merged signal,
    // such as "sin + cos" or similar things, and it can change over time due to drag and drop operations)
    const [streamName, setStreamName] = useState(_streamName_);

    // the whole data plotted in this figure, and array like "emptyPlotData" above, where the "x" and "y" fields grow
    const [allPlotData, setAllPlotData] = useState(emptyPlotData);
    const allPlotDataRef = useRef(allPlotData);

    // largest time step "k" on the whole plot (i.e., the max "k" among all the plotted signals of this figure)
    const [lastStoredStep, setLastStoredStep] = useState(-1);
    const lastStoredStepRef = useRef(lastStoredStep);

    // this is a bare counter, used to trigger the plot update whenever the counter changes (no other usages)
    const [plotUpdatesCounter, setPlotUpdatesCounter] = useState(0);

    // these will only be used to store the values of multiple API calls to stream data, when stream is merged
    const returnedAPICallsRef = useRef(0);
    const updatedLastStoredStepRef = useRef(lastStoredStep);

    // when the stream name changes, it means that some merge operations happened, thus we have to reset the plot
    if (_streamName_ !== streamName) {
        setStreamName(_streamName_);
        setAllPlotData(emptyPlotData);
        setLastStoredStep(-1);
    }
    
    // update reference
    useEffect(() => {
        out("[PlotFigure] useEffect *** update reference: lastStoredStepRef ***");
        lastStoredStepRef.current = lastStoredStep;
    }, [lastStoredStep]);

    // update reference
    useEffect(() => {
        out("[PlotFigure] useEffect *** update reference: allPlotDataRef ***");
        allPlotDataRef.current = allPlotData;
    }, [allPlotData]);

    // fetch data from the stream associated to this figure or from the multiple streams merged into this figure
    useEffect(() => {

        if (!_isPaused_) {
            out("[PlotFigure] useEffect *** fetching data " +
                "(agent_name: " + _agentName_ + ", stream_name (before un-merging): " + _streamName_ + "', " +
                "since_step: " + lastStoredStepRef.current + " *** (skipping, not paused)");
            return;
        }

        // this will tell the parent that this component is working
        _setBusy_(true);

        // declared as an inner-function for readability, it will be used below!
        // store/add new plot data to one of the existing plots of the figure or add a fully new plot
        // (_newPlotDataStorage_ is a (usually empty) map (plotIdx -> plot structure) that gets populated by calling
        // this function several times)
        function storeNewPlotData(_name_, _xData_, _yData_, _newPlotDataStorage_) {
            if (_xData_ == null || _yData_ == null || _xData_.length <= 0 || _yData_.length <= 0) {
                return;
            }
            const numVecComponents = _yData_[0].length;

            // for each vector component on the _yData_ ...
            for (let vecComponentIdx = 0; vecComponentIdx < Math.min(numVecComponents, maxVecComponentsPerSignal);
                 vecComponentIdx++) {

                // check if we reached the max plot size
                if (allPlotDataRef.current.length + _newPlotDataStorage_.size >= maxTotalPlots) {
                    break;
                }

                // for vectorial data the name of the plot is "plotName~0", "plotName~1", ...
                const plotName = numVecComponents > 1 ? _name_ + "~" + vecComponentIdx : _name_;

                // new plots has plotIdx equal to -1, while existing plots have plotIdx >= 0
                // (plots are found comparing the names of the plot)
                let plotIdx = allPlotDataRef.current.findIndex(plot => plot.name === plotName);

                if (plotIdx === -1) {

                    // new plot with its own plotIdx (the negative index was temporary)
                    plotIdx = -_newPlotDataStorage_.size;

                    // this map is plotIdx -> plot data struct (see the "emptyPlotData" at the top of this file)
                    _newPlotDataStorage_.set(plotIdx, {
                        x: _xData_,
                        y: _yData_.map(my_y_data_array => my_y_data_array[vecComponentIdx]), // keep only 1 component
                        type: "scatter",
                        mode: "lines",
                        name: plotName
                    });
                } else {

                    // augmenting an existing plot, for which there was already a mapping plotIdx -> plot in the map
                    _newPlotDataStorage_.set(plotIdx, {
                        new_x: _xData_,
                        new_y: _yData_.map(my_y_data_array => my_y_data_array[vecComponentIdx]) // keep only 1 component
                    });
                }
            }
        }

        // fetching data from a single stream or, if merged, from multiple streams
        const unmergedStreamNames = _streamName_.split(' + ').map(item => item.trim());
        const numMergedStreams = unmergedStreamNames.length;
        const newPlotDataStorage = new Map(); // created as empty map, populated by storeNewPlotData(...)
        returnedAPICallsRef.current = 0; // we will count how many of the merged stream return data and what fails
        updatedLastStoredStepRef.current = lastStoredStepRef.current;

        for (let j = 0; j < numMergedStreams; j++) {

            out("[PlotFigure] useEffect *** fetching stream data " + (j+1) + "/" + numMergedStreams + " " +
                "(agent_name: " + _agentName_ + ", stream_name: " + unmergedStreamNames[j] + "', " +
                "since_step: " + lastStoredStepRef.current);

            callAPI('/get_stream', {
                    agent_name: _agentName_,
                    stream_name: unmergedStreamNames[j],
                    since_step: lastStoredStepRef.current + 1
                },
                (x) => {

                    // initial number of new plots temporarily buffered in the storage map
                    const numNewPlotsCurrentlyStored = newPlotDataStorage.size;

                    // here we store the received data into the temporary storage
                    storeNewPlotData(unmergedStreamNames[j], x.ks, x.data, newPlotDataStorage);

                    // if the temporary map size did not change, we reached the max number of plots and skipped this one
                    if (newPlotDataStorage.size > numNewPlotsCurrentlyStored) {
                        updatedLastStoredStepRef.current = Math.max(x.last_k, updatedLastStoredStepRef.current);
                    }

                    // we actually change the real-plot-object data using the temporarily stored plots
                    // only when getting data from the API call about the last merged stream (if one fails, we do not
                    // update anything, since we will never reach the numMergedStreams number)
                    returnedAPICallsRef.current++;

                    // when reaching the last stream...
                    if (returnedAPICallsRef.current === numMergedStreams && newPlotDataStorage.size > 0) {

                        // here we change the real-plot-object data, either adding new plots or augmenting others
                        setAllPlotData((prevAllPlotData) => {
                            const curAllPlotData = [...prevAllPlotData]; // clone

                            for (const [plotIdx, plotData] of newPlotDataStorage.entries()) {
                                if (plotIdx >= 0 && curAllPlotData[plotIdx]) { // existing plot: augment it
                                    curAllPlotData[plotIdx].x.push(...plotData.new_x);  // append new
                                    curAllPlotData[plotIdx].y.push(...plotData.new_y);  // append new y
                                } else {
                                    curAllPlotData.push(plotData); // new plot: add it
                                }
                            }

                            // purging streams that are not part of this figure anymore (due to unmerging)
                            // and returning the current "purged" data
                            return curAllPlotData.filter((plotDataStruct) => {
                                const delimiterIndex = plotDataStruct.name.lastIndexOf("~");
                                const _stream_name = delimiterIndex !== -1 ?
                                    plotDataStruct.name.substring(0, delimiterIndex) : plotDataStruct.name;
                                return unmergedStreamNames.includes(_stream_name);
                            });
                        });

                        // save the largest x-value, it will be used in the future to ask for new data
                        setLastStoredStep((prevStep) =>
                            Math.max(prevStep, updatedLastStoredStepRef.current));

                        // this will trigger an update of the plot graphic
                        // (Plotly is listening to changes to this counter)
                        setPlotUpdatesCounter((prev) => prev + 1);
                    }
                },
                () => {
                    setLastStoredStep((prev) => (prev));
                    setAllPlotData((prevData) => (prevData));
                },
                () => {
                    if (returnedAPICallsRef.current === numMergedStreams) {
                        // this will tell the parent that this component is now ready
                        _setBusy_(false);
                    }
                }
            );
        }
    }, [_isPaused_, _streamName_, _agentName_, _setBusy_]);
    // _isPaused_ is what we care, while _streamName_ changes due to merge/unmerge (_agentName_, _setBusy_ are constant)

    // plot layout (Plotly)
    const layout = {
        plot_bgcolor: 'rgba(0, 0, 0, 0)', // transparent plot area
        paper_bgcolor: 'rgba(0, 0, 0, 0)', // transparent entire figure
        margin: { t: 0, b: 'auto', l: 40, r: 40 }, // reduce margins
        xaxis: {
            title: "Time",
            type: "category", // this is fine when the x-axis component are explicitly provided
        },
        yaxis: {
            title: "Amplitude",
        },
        legend: {
            orientation: 'h', // horizontal legend
            x: 0.5, // centered horizontally
            y: 1.05, // position slightly above the plot
            xanchor: 'center', // align to center horizontally
            yanchor: 'bottom', // align to bottom of the legend container
            font: {
                size: 12,
                family: 'Arial, sans-serif'
            },
        },
        dragmode: "zoom", // allow zooming and panning
        autosize: true
        //width: 600, // not currently using this, kept as example
        //height: 500 // not currently using this, kept as example
    };

    // returning the <div>...</div> that will be displayed
    return (
        <div style={{width: "100%", height: "100%"}}>
            <Plot
                key={plotUpdatesCounter + _streamName_} // changes in plotUpdatesCounter and _streamName_ trigger update
                data={allPlotData}  // data about all the plots in this figure
                layout={layout}  // see above
                config={{
                    responsive: true, // this might be unhappy with "autosize", but let's see
                    displayModeBar: false, // turning off the (nice) Plotly display bar
                }}
                style={{width: '100%', height: '100%'}}
                useResizeHandler
            />
        </div>
    );
}
