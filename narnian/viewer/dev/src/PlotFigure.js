import {useRef, useState, useEffect} from "react";
import Plot from "react-plotly.js";
import {callAPI, out} from "./utils";

const maxTotalPlots = 8; // customizable (total number of plots in the same figure)
const maxVecComponentsPerSignal = 6; // customizable (max number of components from a vectorial signal)
const id2Color = { // plot colors
    0: "#1F77B4", // light blue
    1: "#FF7F0E", // orange
    2: "#2CA02C", // green
    3: "#D62728", // red
    4: "#9467BD", // purple
    5: "#8C564B", // brown
    6: "#E377C2", // pink
    7: "#7F7F7F", // gray
    8: "#BCBD22", // olive
    9: "#17BECF" // cyan
}

export default function PlotFigure({ _agentName_, _streamStruct_, _isPaused_, _setBusy_ }) {
    out("[PlotFigure] " +
        "_agentName_: " + _agentName_ + ", " +
        "_streamStruct_: " + JSON.stringify(_streamStruct_) + ", " +
        "_isPaused_: " + _isPaused_);

    // basic structure of the data that will be plotted: it is an array where each element is the data of a single plot
    // in particular, each element is {x: [...], y: [...], type: "scatter", mode: "lines", name: "plot-name"}
    const emptyXYs = Array.from({
            length: 0
        }, (_,) => ({
            x: [],
            y: [],
            type: "scatter",
            mode: "lines",
            line: { color: id2Color[0] },
            name: null,
        })
    );

    // basic structure taking case of xLims and yLims
    const emptyMIMAXs = {xMin: 0, yMin: 0, xMax: 0, yMax: 0}

    // this is the basic structure of ALL the data that will be used py the plot
    const emptyPlotData = {
        XYs: emptyXYs,
        PNGs: [],
        TEXTs: [],
        MIMAXs: emptyMIMAXs,
        limitToLastN: false,
        version: 0,
    }

    // the name, ids, of the streams in this figure (it can be about a single or a merged signal,
    // and it can change over time due to drag and drop operations)
    const [streamStruct, setStreamStruct] = useState(_streamStruct_);

    // the whole data plotted in this figure, and array like "emptyPlotData" above, where the "x" and "y" fields grow
    const XYs = useRef(emptyXYs);
    const PNGs = useRef([]);
    const TEXTs = useRef([]);
    const MIMAXs = useRef(emptyMIMAXs);
    const limitToLastN = useRef(false);
    const [allPlotData, setAllPlotData] = useState(emptyPlotData);

    // these will only be used to store the values of multiple API calls to stream data, when stream is merged
    const returnedAPICallsRef = useRef(0);

    // when the stream struct changes, it means that some merge operations happened, thus we have to reset the plot
    if (_streamStruct_.mergedIds.length !== streamStruct.mergedIds.length ||
        !(_streamStruct_.mergedIds.every((value, index) =>
            value === streamStruct.mergedIds[index]))) {
        setStreamStruct(_streamStruct_);
        setAllPlotData(emptyPlotData);
    }

    // update reference
    useEffect(() => {
        out("[PlotFigure] useEffect *** update reference to the fields of: allPlotDataRef ***");
        XYs.current = allPlotData.XYs;
        PNGs.current = allPlotData.PNGs;
        TEXTs.current = allPlotData.TEXTs;
        MIMAXs.current = allPlotData.MIMAXs;
    }, [allPlotData]);

    // fetch data from the stream associated to this figure or from the multiple streams merged into this figure
    useEffect(() => {

        if (!_isPaused_) {
            out("[PlotFigure] useEffect *** fetching data " +
                "(agent_name: " + _agentName_ + ", stream_name (before un-merging): " + _streamStruct_.label + "', " +
                "since_step: " + (MIMAXs.current.xMax + 1) + " *** (skipping, not paused)");
            return;
        }

        // this will tell the parent that this component is working
        _setBusy_(true);

        // declared as an inner-function for readability, it will be used below!
        // store/add new plot data to one of the existing plots of the figure or add a fully new plot
        // (_nexXYsStorage_ is a (usually empty) map (plotIdx -> plot structure) that gets populated by calling
        // this function several times)
        function storeNewPlotData(_colorId_, _name_, _xData_, _yData_,
                                  _nexXYsStorage_, _newPNGsStorage_, _newTEXTsStorage_) {
            if (_xData_ == null || _yData_ == null || _xData_.length <= 0 || _yData_.length <= 0) {
                return;
            }

            const PNGDetected = (typeof _yData_[0] === 'string' && _yData_[0].startsWith("data:image/png;base64"))
            const textDetected = (typeof _yData_[0] === 'string' && !_yData_[0].startsWith("data:image/png;base64"))
            const numVecComponents = PNGDetected || textDetected ? 1 : _yData_[0].length;

            // for each vector component on the _yData_ ...
            for (let vecComponentIdx = 0;
                 vecComponentIdx < Math.min(numVecComponents, maxVecComponentsPerSignal);
                 vecComponentIdx++) {

                // check if we reached the max plot size
                if (XYs.current.length + _nexXYsStorage_.size >= maxTotalPlots) {
                    break;
                }

                // for vectorial data the name of the plot is "plotName~0", "plotName~1", ...
                const plotName = numVecComponents > 1 ? _name_ + "~" + vecComponentIdx : _name_;

                // new plots has plotIdx equal to -1, while existing plots have plotIdx >= 0
                // (plots are found comparing the names of the plot)
                let plotIdx = XYs.current.findIndex(plot => plot.name === plotName);

                if (plotIdx === -1) {

                    // new plot with its own plotIdx (the negative index was temporary)
                    plotIdx = _nexXYsStorage_.size;

                    // this map is plotIdx -> plot data struct (see the "emptyPlotData" at the top of this file)
                    _nexXYsStorage_.set(plotIdx, {
                        x: _xData_,
                        y: !PNGDetected && !textDetected ?
                            _yData_.map(my_y_data_array => my_y_data_array[vecComponentIdx])  // keep only 1 component
                            : Array(_xData_.length).fill(plotIdx),  // plot PNG images and text annotations at y=plotIdx
                        type: "scatter",
                        mode: "lines",
                        line: { color: id2Color[_colorId_] },
                        name: plotName
                    });
                } else {

                    // augmenting an existing plot, for which there was already a mapping plotIdx -> plot in the map
                    _nexXYsStorage_.set(plotIdx, {
                        new_x: _xData_,
                        new_y: !PNGDetected && !textDetected ?
                            _yData_.map(my_y_data_array => my_y_data_array[vecComponentIdx])  // keep only 1 component
                            : Array(_xData_.length).fill(plotIdx),  // plot PNG images and text annotations at y=plotIdx
                    });
                }

                // this is the set of PNG images received by the single/multiple call(s) to get_stream
                if (PNGDetected) {
                    _newPNGsStorage_.push(..._yData_.map((pngImageBase64, index) => ({
                        source: pngImageBase64, // base64 image source
                        x: _xData_[index],
                        y: plotIdx + 0.025, // fixed
                        sizex: 0.95, // size of the image in x-direction
                        sizey: 0.95, // size of the image in y-direction
                        xanchor: "center",
                        yanchor: "bottom", // "middle",
                        layer: "above", // ensure images are on top of markers
                        xref: 'x', // use the x-axis scale (data coordinates)
                        yref: 'y' // use the y-axis scale (data coordinates)
                    })));

                    limitToLastN.current = true; // this marks that we want to see only a small set of recent samples

                // this is the set of text annotations (words) received by the single/multiple call(s) to get_stream
                } else if (textDetected) {
                    _newTEXTsStorage_.push(..._yData_.map((textAnnotation, index) => ({
                        text: textAnnotation, // text
                        x: _xData_[index],
                        y: plotIdx, // fixed
                        showarrow: true,
                        font: {
                            family: 'Arial, sans-serif',
                            size: 14,
                            color: 'black',
                            //weight: 'bold'
                        },
                        xanchor: "center",
                        yanchor: "middle",
                    })));

                    limitToLastN.current = true; // this marks that we want to see only a small set of recent samples
                }
            }
        }

        // fetching data from a single stream or, if merged, from multiple streams
        const streamIDs = _streamStruct_.mergedIds;
        const streamNames = _streamStruct_.mergedLabels;
        const numStreams = streamIDs.length;
        const nexXYsStorage = new Map(); // created as empty map, populated by storeNewPlotData(...)
        const newPNGsStorage = []; // created as empty array, populated by storeNewPlotData(...)
        const newTEXTsStorage = []; // created as empty array, populated by storeNewPlotData(...)
        returnedAPICallsRef.current = 0; // we will count how many of the merged stream return data and what fails
        limitToLastN.current = false;

        for (let j = 0; j < numStreams; j++) {

            out("[PlotFigure] useEffect *** fetching stream data " + (j+1) + "/" + numStreams + " " +
                "(agent_name: " + _agentName_ + ", stream_name: " + streamNames[j] + "', " +
                "since_step: " + (MIMAXs.current.xMax + 1));

            callAPI('/get_stream', {
                    agent_name: _agentName_,
                    stream_name: streamNames[j],
                    since_step: MIMAXs.current.xMax + 1
                },
                (x) => {

                    // here we store the received data into the temporary storage
                    storeNewPlotData(j, streamNames[j], x.ks, x.data,
                        nexXYsStorage, newPNGsStorage, newTEXTsStorage);

                    // we actually change the real-plot-object data using the temporarily stored plots
                    // only when getting data from the API call about the last merged stream (if one fails, we do not
                    // update anything, since we will never reach the numStreams number)
                    returnedAPICallsRef.current++;

                    // when reaching the last stream...
                    if (returnedAPICallsRef.current === numStreams && nexXYsStorage.size > 0) {

                        // here we change the real-plot-object data, either adding new plots or augmenting others
                        for (const [plotIdx, plotData] of nexXYsStorage.entries()) {
                            if (plotIdx >= 0 && XYs.current[plotIdx]) { // existing plot: augment it
                                XYs.current[plotIdx].x.push(...plotData.new_x);  // append new
                                XYs.current[plotIdx].y.push(...plotData.new_y);  // append new y
                            } else {
                                XYs.current.push(plotData); // new plot: add it
                            }
                        }

                        // purging streams that are not part of this figure anymore (due to unmerging)
                        // and returning the current "purged" data
                        XYs.current = XYs.current.filter((plotDataStruct) => {
                            const delimiterIndex = plotDataStruct.name.lastIndexOf("~");
                            const _streamName = delimiterIndex !== -1 ?
                                plotDataStruct.name.substring(0, delimiterIndex) : plotDataStruct.name;
                            return streamNames.includes(_streamName);
                        });

                        // estimating min and max of the whole data
                        XYs.current.forEach(trace => {
                            const xValues = trace.x;
                            const yValues = trace.y;
                            MIMAXs.current.xMin = Math.min(...xValues);
                            MIMAXs.current.xMax = Math.max(...xValues);
                            MIMAXs.current.yMin = Math.min(...yValues);
                            MIMAXs.current.yMax = Math.max(...yValues);
                        });

                        // here we augment the current set of images with the newly received ones
                        if (newPNGsStorage.length > 0) {

                            // we assume PNG plots to be at y-coordinates that are 0, 1, 2... and we assume they are
                            // "tall" 1.0
                            MIMAXs.current.yMax = Math.max(MIMAXs.current.yMax, XYs.current.length - 1 + 1.0);
                            MIMAXs.current.yMin = Math.min(MIMAXs.current.yMin, 0.);
                            PNGs.current.push(...newPNGsStorage); // add newly received PNGs
                        }

                        // here we augment the current set of text annotations with the newly received ones
                        if (newTEXTsStorage.length > 0) {

                            // we assume PNG plots to be at y-coordinates that are 0, 1, 2... and we assume they are
                            // "tall" 1.0
                            MIMAXs.current.yMax = Math.max(MIMAXs.current.yMax, XYs.current.length - 1 + 1.0);
                            MIMAXs.current.yMin = Math.min(MIMAXs.current.yMin, 0.);
                            TEXTs.current.push(...newTEXTsStorage); // add newly received PNGs
                        }

                        // now we update the state
                        setAllPlotData((prev) => {
                            return {
                                XYs: XYs.current,
                                PNGs: PNGs.current,
                                TEXTs: TEXTs.current,
                                MIMAXs: MIMAXs.current,
                                limitToLastN: limitToLastN.current,
                                version: prev.version + 1
                            }
                        });
                    }
                },
                () => {
                    setAllPlotData((prevData) => (prevData));
                },
                () => {
                    if (returnedAPICallsRef.current === numStreams) {
                        // this will tell the parent that this component is now ready
                        _setBusy_(false);
                    }
                }
            );
        }
    }, [_isPaused_, _streamStruct_, _agentName_, _setBusy_]);
    // _isPaused_ is what we care, while _streamStruct_ changes due to (un)merge (_agentName_, _setBusy_ are constant)

    // returning the <div>...</div> that will be displayed
    return (
        <div style={{width: "100%", height: "100%"}}>
            <Plot
                key={allPlotData.version}
                data={allPlotData.XYs}  // data about all the plots in this figure
                layout={{
                    plot_bgcolor: 'rgba(0, 0, 0, 0)', // transparent plot area
                    paper_bgcolor: 'rgba(0, 0, 0, 0)', // transparent entire figure
                    margin: {t: 0, b: 'auto', l: 40, r: 40}, // reduce margins
                    xaxis: {
                        range: [!allPlotData.limitToLastN ? allPlotData.MIMAXs.xMin : allPlotData.MIMAXs.xMax - 5.5,
                            !allPlotData.limitToLastN ? allPlotData.MIMAXs.xMax : allPlotData.MIMAXs.xMax + 0.5],
                        title: "Time",
                        type: "linear", // this is fine when the x-axis component are explicitly provided
                    },
                    yaxis: {
                        range: [allPlotData.MIMAXs.yMin,
                            !allPlotData.limitToLastN ? allPlotData.MIMAXs.yMax : allPlotData.MIMAXs.yMax + 1.0],
                        title: "",
                        type: "linear",
                    },
                    images: allPlotData.PNGs, // array of images, if any
                    annotations: allPlotData.TEXTs, // array of textual annotations (e.g., words), if any
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
                }}  // see above
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
