import {useEffect, useRef, useState} from "react";
import { select as d3_select, zoom as d3_zoom, zoomIdentity as d3_zoomIdentity, interrupt as d3_interrupt } from "d3";
import { forceSimulation as d3_forceSimulation, forceLink as d3_forceLink, forceManyBody as d3_forceManyBody,
    forceCenter as d3_forceCenter, forceCollide as d3_forceCollide} from "d3-force";
import { drag as d3_drag } from "d3-drag";
import { read as dotRead } from "graphlib-dot"
import {callAPI, out} from "./utils"

export default function FSM({_agentName_, _isPaused_, _setBusy_}) {
    out("[FSM] " +
        "_agentName_: " + _agentName_ + ", " +
        "_isPaused_: " + _isPaused_);

    // reference to the SVG area where the graph will de displayed
    const svgRef = useRef();

    // references to the node or edge that are currently highlighted (or zoomed in) and to the zoom transition
    const highlightedNodeRef = useRef(null);
    const highlightedEdgeRef = useRef(null);
    const activeZoomTransitionRef = useRef(null);

    // state of the SVG area (structured as {width: a, height: b})
    const [svgSize, setSvgSize] = useState({width: 0, height: 0});
    const svgSizeRef = useRef(svgSize);

    // flag to recall when the data is being loaded through the API call
    const [loading, setLoading] = useState(true);

    // flag to signal that drawing operation has ended
    const [doneDrawing, setDrawingDone] = useState(false);

    // data in GraphViz format (string)
    const [graphvizDotStringData, setGraphvizDotStringData] = useState(null);

    // the D3 simulation engine
    const simulationRef = useRef(null);

    // fetch the GraphViz string of the FSM
    useEffect(() => {
        out("[FSM] useEffect *** fetching data (get behaviour, agent name: " + _agentName_ + ") ***");
        setLoading(true); // loading started

        callAPI('/get_behav', "agent_name=" + _agentName_,
            (x) => {
                setGraphvizDotStringData(x);
            },
            () => { setGraphvizDotStringData(null); return true; },  // clearing
            () => {
                setLoading(false);  // done loading
            })
    }, [_agentName_]);  // _agentName_ is not going to change

    // fetch the current state/action
    useEffect(() => {
        if (!_isPaused_) {
            out("[FSM] useEffect *** fetching data (get behaviour status, agent name: " + _agentName_ + ") *** " +
                "(skipping, not paused)");
            return;
        }

        if (!doneDrawing) {
            out("[FSM] useEffect *** fetching data (get behaviour status, skipping - too early) ***");
            return;
        }

        out("[FSM] useEffect *** fetching data (get behaviour status, agent name: " + _agentName_ + ") ***");

        callAPI('/get_behav_status', "agent_name=" + _agentName_,
            (x) => {
                if (x.state != null && x.action == null) {
                    const node2Highlight = "node" + x.state;

                    out("[FSM] Highlighting (clicking on) node '" + node2Highlight + "'");
                    const targetNode = d3_select(svgRef.current).select("#" + node2Highlight);

                    if (!targetNode.empty()) {
                        targetNode.dispatch("click"); // dispatching a "click" event on node
                    } else {
                        out("[FSM] Node with id '" + node2Highlight + "' not found");
                    }
                } else if (x.action != null && x.state == null) {
                    const edge2Highlight = "edge" + x.action;

                    out("[FSM] Highlighting edge '" + edge2Highlight + "'");
                    const targetEdge = d3_select(svgRef.current).select("#" + edge2Highlight);

                    if (!targetEdge.empty()) {
                        targetEdge.dispatch("click"); // dispatching a "click" event on edge
                    } else {
                        out("[FSM] Edge with id '" + edge2Highlight + "' not found");
                    }
                } else {
                    throw new Error("Unknown behaviour status (state: " + x.state + ", action: " + x.action + ")");
                }
            },
            () => {
                return true;
            },
            () => {
            }
        );
    }, [doneDrawing, _isPaused_, _agentName_]);
    // listens to doneDrawing (when drawing is done, it's time to highlight) and _isPaused_

    // resize the FSM drawing
    useEffect(() => {
        if (!graphvizDotStringData) {
            out("[FSM] useEffect *** resizing *** (skipping, graph not loaded yet)");
            return;
        }

        out("[FSM] useEffect *** resizing ***");

        const handleResize = () => {
            if (svgRef.current) {
                const {width, height} = svgRef.current.getBoundingClientRect();
                out("[FSM] handleResize -> setSvgSize to width: " + width + ", height: " + height);
                if (svgSizeRef.current.width === 0 && svgSizeRef.current.height === 0) {
                    setSvgSize(() => {
                            const newSize = {width, height};
                            svgSizeRef.current = newSize;
                            return newSize;
                        }
                    );
                }
            }
        };

        // listen to the resize event and trigger handleResize above in case of event
        window.addEventListener('resize', handleResize);

        // resize on CSS property changes using ResizeObserver
        const resizeObserver = new ResizeObserver(handleResize);
        if (svgRef.current) {
            out("[FSM] Adding ResizeObserver");
            resizeObserver.observe(svgRef.current);
        }

        // initial resize call
        out("[FSM] calling handleResize as initial setter");
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);  // clearing event
            resizeObserver.disconnect(); // clean up ResizeObserver
        };
    }, [graphvizDotStringData]); // when data has been loaded, it's time to resize

    // drawing operations (when the data is fully loaded and when the size changes)
    useEffect(() => {
        if (!graphvizDotStringData) {
            out("[FSM] useEffect *** drawing *** (graphviz data not ready yet, skipping)");
            return;
        }
        if (svgSize.width <= 0 || svgSize.height <= 0) {
            out("[FSM] useEffect *** drawing *** (size too small, skipping)");
            return;
        }

        // this will tell the parent that this component is working
        _setBusy_((prev) => prev + 1);

        const {width, height} = svgSize;
        out("[FSM] useEffect *** drawing *** (data ready, width: " + width + ", height: " + height + ")");

        setDrawingDone(false);  // marking the drawing has started

        const updateNodeHighlighting = () => {
            node.attr("fill", (d) => (d.id === highlightedNodeRef.current?.id ? "#82e0aa" : "#aed6f1"));
            label.style("font-weight", (d) => (d.id === highlightedNodeRef.current?.id ? "bold" : "normal"));
        };

        const updateEdgeHighlighting = () => {
            link.attr("stroke", (d) => (d === highlightedEdgeRef.current ? "#82e0aa" : "#b2babb"));
            link.attr("marker-end", (d) =>
                d === highlightedEdgeRef.current ? "url(#arrow-highlighted)" : "url(#arrow)");
            edgeLabel.style("font-weight", (d) => (d === highlightedEdgeRef.current ? "bold" : "normal"));
        };

        const zoomToEdge = (edge) => {
            const centerX = (edge.source.x + edge.target.x) / 2;
            const centerY = (edge.source.y + edge.target.y) / 2;
            const scale = 1.3;
            d3_interrupt(svg, activeZoomTransitionRef.current);  // stopping other possibly running transitions (zooms)
            activeZoomTransitionRef.current = svg.transition().duration(750)
                .call(zoom.transform, d3_zoomIdentity.translate(width / 2, height / 2)
                    .scale(scale).translate(-centerX, -centerY));
        };

        const zoomToNode = (node) => {
            const scale = 1.3;
            d3_interrupt(svg, activeZoomTransitionRef.current);  // stopping other possibly running transitions (zooms)
            activeZoomTransitionRef.current = svg.transition().duration(750)
                .call(zoom.transform, d3_zoomIdentity.translate(width / 2, height / 2)
                    .scale(scale).translate(-node.x, -node.y));
        };

        const highlightEdge = (d) => {
            highlightedEdgeRef.current = d;
            highlightedNodeRef.current = null;
            updateNodeHighlighting();
            updateEdgeHighlighting();
            zoomToEdge(d);
        }

        const highlightNode = (d) => {
            highlightedNodeRef.current = d;
            highlightedEdgeRef.current = null;
            updateNodeHighlighting();
            updateEdgeHighlighting();
            zoomToNode(d);
        }

        // getting the SVG area (by means of D3)
        const svg = d3_select(svgRef.current)
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", `0 0 ${width} ${height}`)
            .attr("preserveAspectRatio", "xMidYMid meet");

        // clearing the whole SVG area
        svg.selectAll("*").remove();

        // appending a new element to the SVG area, which is a placeholder for the graph
        const g = svg.append("g");

        // zooming to get the right size
        const zoom = d3_zoom().on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);

        // reading the GraphViz string into graph named "graph"
        const graph = dotRead(graphvizDotStringData);

        // from GraphViz "graph": creating a node structure that includes the IDs that were manually added in GraphViz
        const nodesWithIDs = graph.nodes().map(nodeName => {
            const attributes = graph.node(nodeName);
            return {
                name: nodeName,
                id: attributes.id,
                label: attributes.label.trim()
            };
        });

        // saving a map "name of the node (string)" -> "manually added ID of the node" (assuming it is invertible!)
        const nodeNameToNodeID = Object.fromEntries(nodesWithIDs.map(node => [node.name, node.id]));

        // from GraphViz "graph": creating an edge structure that includes the IDs that were manually added in GraphViz
        const edgesWithIDs = graph.edges().map(edge => {
            const attributes = graph.edge(edge.v, edge.w, edge.name);
            return {
                name: edge.name,
                source: nodeNameToNodeID[edge.v],
                target: nodeNameToNodeID[edge.w],
                label: attributes.label,
                id: attributes.id
            };
        });

        // creating a graph structure based on the just created node/edge structures, fully discarding GraphViz "graph"
        const graphData = {
            nodes: nodesWithIDs,
            links: edgesWithIDs
        };

        // creating a new simulation (and stopping it, immediately)
        const sim = d3_forceSimulation(graphData.nodes)
            .force("link", d3_forceLink(graphData.links).id((d) => d.id).distance(100))
            .force("charge", d3_forceManyBody().strength(-150))
            .force("center", d3_forceCenter(width / 2, height / 2))
            .force("collide", d3_forceCollide().radius(100))
            .stop();

        // saving the state with the new simulation (and possibly stopping an already running one)
        if (simulationRef.current) {
            out("[FSM] stopping previous simulation");
            simulationRef.current.stop();
        }
        simulationRef.current = sim;

        // manually running the simulation until this nice condition is met (or try for 300 steps)
        while (sim.alpha() > sim.alphaMin()) {
            sim.tick();
        }

        // Function to generate the "d" attribute for curved edges
        function getLinkPath(d) {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy) * 1.5; // Control radius for rounded edges
            return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
        }

        // adding edges to the SVG-graph "g", based on the pre-processed GraphViz edges in "graphData"
        const link = g.selectAll("path")
            .data(graphData.links)
            .enter()
            .append("path")
            .attr("stroke", "#34495e")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr("marker-end", "url(#arrow)")
            .attr("stroke", (d) => (d === highlightedEdgeRef.current ? "#e74c3c" : "#b2babb"))
            .style("pointer-events", "none")
            .attr("d", getLinkPath)
            .on("click", (event, d) => {
                event.stopPropagation();
                if (!event.isTrusted) {
                    highlightEdge(d);
                }
            });

        // adding edge labels to the SVG-graph "g", based on the pre-processed GraphViz edges in "graphData"
        const edgeLabel = g.selectAll("text.edge-label")
            .data(graphData.links)
            .enter()
            .append("text")
            .attr("class", "edge-label")
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .attr("x", (d) => (0.3 * d.source.x + 0.7 * d.target.x))
            .attr("y", (d) => (0.6 * d.source.y + 0.4 * d.target.y))
            .attr("id", (d) => (d.id))
            .style("cursor", "default")
            .on("click", (event, d) => {
                event.stopPropagation();
                if (!event.isTrusted) {
                    highlightEdge(d);
                }
            }).each(function (d) {
                const label = d3_select(this);
                const pos = d.label.indexOf("(");
                const mainText = d.label.slice(0, pos);
                const subText = (pos === (d.label.length - 3)) ? "" : d.label.slice(pos);

                const dy = d.target.y - d.source.y;
                const relationLabelY = subText.length > 0 ? (-3 + (dy > 0 ? 5 : -5)) : (5 + (dy > 0 ? 5 : -5));

                label.append("tspan")
                    .text(mainText)
                    .style("font-size", "12px")
                    .attr("x", d3_select(this).attr("x"))
                    .attr("dy", relationLabelY.toString());

                if (subText.length > 0) {
                    const subTexts = [];
                    const maxLength = 35;
                    for (let i = 0; i < subText.length; i += maxLength) {
                        subTexts.push(subText.slice(i, i + maxLength));
                    }
                    subTexts.forEach((text, index) => {
                        label.append("tspan")
                            .text(text)
                            .style("font-size", "6px")
                            .attr("x", d3_select(this).attr("x"))
                            .attr("dy", index === 0 ? "7" : "6");
                    });
                }
            });

          // dynamically update the width and height of the node based on the label's bounding box (used many times)
        function updateNodePositionsAndSizes() {

            // update nodes
            node.each(function (d) {
                const textElement = label.filter((ld) => ld.id === d.id).node();
                if (textElement) {
                    const bbox = textElement.getBBox();
                    d.width = bbox.width + 20;
                    d.height = bbox.height + 12;
                }
            }).attr("width", (d) => d.width)
                .attr("height", (d) => d.height)
                .attr("x", (d) => d.x - d.width / 2)
                .attr("y", (d) => d.y - d.height / 2);

            // update also edge positions
            link.attr("d", getLinkPath);

            // update also edge labels' positions
            edgeLabel.attr("x", (d) => (0.3 * d.source.x + 0.7 * d.target.x))
                .attr("y", (d) => (0.6 * d.source.y + 0.4 * d.target.y))
                .each(function (d) {
                    const label = d3_select(this);
                    label.selectAll("tspan").attr("x", d3_select(this).attr("x"));
                });

            // update node labels' positions to match the node's center
            label.attr("x", d => d.x)
                .attr("y", d => d.y)
                .each(function (d) {
                    const label = d3_select(this);
                    label.selectAll("tspan").attr("x", d3_select(this).attr("x"));
                });
        }

        // drag action, applied to nodes (will be provided as attribute in what follows)
        const drag = d3_drag()
            .on("start", (event, d) => {

                // fixing position of nodes and labels (to synch them at the beginning)
                d.fx = d.x;
                d.fy = d.y;
            })
            .on("drag", (event, d) => {

                // updating node position in the event data structure (not on the node yet)
                d.fx = event.x;
                d.fy = event.y;
                d.x = event.x;
                d.y = event.y;

                // running a simulation tick to let the other nodes/edges move in function of the current movement
                sim.tick();

                // dynamically update the width and height of the node based on the label's bounding box
                updateNodePositionsAndSizes();
            })
            .on("end", (event, d) => {

                // when drag ends, we do the same thing done above
                updateNodePositionsAndSizes();

                // adding a weird condition on the simulation
                if (!event.active) {
                    sim.alphaTarget(0.3);
                }

                // unfix after dragging
                d.fx = null;
                d.fy = null;
            });

        // adding nodes to the SVG-graph "g", based on the pre-processed GraphViz edges in "graphData"
        const node = g.selectAll("rect")
            .data(graphData.nodes)
            .enter()
            .append("rect")
            .attr("rx", 12)
            .attr("ry", 12)
            .attr("id", d => d.id)
            .style("pointer-events", "auto")
            .attr("fill", (d) => (d === highlightedNodeRef.current ? "#82e0aa" : "#aed6f1"))
            .on("click", (event, d) => {
                event.stopPropagation();
                if (!event.isTrusted)
                    highlightNode(d);
            })
            .call(drag);

        // adding node labels to the SVG-graph "g", based on the pre-processed GraphViz edges in "graphData"
        const label = g.selectAll("text.node-label")
            .data(graphData.nodes)
            .enter()
            .append("text")
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("font-size", "14px")
            .style("pointer-events", "none")
            .each(function (d) {
                const label = d3_select(this);
                const pos = d.label.indexOf("\n");
                const mainText = (pos > 0) ? d.label.slice(0, pos).trim() : d.label;
                const subText = (pos > 0) ? d.label.slice(pos + 1).trim() : null;

                label.append("tspan")
                    .text(mainText)
                    .style("font-size", "14px")
                    .attr("x", d3_select(this).attr("x"))
                    .attr("dy", (subText) ? "-2" : "0");

                if (subText) {
                    label.append("tspan")
                        .text(subText)
                        .style("font-size", "6px")
                        .attr("x", d3_select(this).attr("x"))
                        .attr("dy", "9");
                }
            });

        // dynamically update the width and height of the node based on the label's bounding box
        updateNodePositionsAndSizes();

        // adding marker defs to the SVG-graph "g"
        const defs = g.append("defs");

        // arrow marker
        defs.append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#b2babb");

        // highlighted arrow marker
        defs.append("marker")
            .attr("id", "arrow-highlighted")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#82e0aa");

        // when clicking on the SVG-area (not on the graph), a de-zoom operation is triggered
        svg.on("click", () => {
            svg.transition().duration(750).call(zoom.transform, d3_zoomIdentity); // reset zoom to original
        });

        // for every simulation tick, we have to update the position of the nodes (that changes!)
        sim.on("tick", () => {
            node.attr("x", d => d.x).attr("y", d => d.y);
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            g.selectAll(".node-label")
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });

        // if this drawing operation occurred due to a resize and there were already nodes/edges highlighted, we
        // ensure they get re-highlighted
        if (highlightedNodeRef.current != null) {
            highlightNode(highlightedNodeRef.current);
        } else if (highlightedEdgeRef.current != null) {
            highlightEdge(highlightedEdgeRef.current);
        }

        out("[FSM] Graph simulation/drawing ended");
        setDrawingDone(true);  // marking the drawing has completed

        // this will tell the parent that this component is now ready
        _setBusy_((prev) => prev - 1);
    }, [graphvizDotStringData, svgSize, _setBusy_]);
    // redraw when the data is loaded and when the size changes due to resize (_setBusy_ will not change)

    // returning the <div>...</div> that will be displayed when loading data (rotating spin or similar)
    if (loading) {
        return (
            <div className="flex items-center justify-center w-full h-full">
                <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"/>
            </div>
        )
    }

    // returning the <div>...</div> that will be displayed
    return (
        <div className="flex items-center justify-center w-full h-full">
            <svg ref={svgRef} className="w-full h-full p-0 pb-1"/>
        </div>
    );
}
