import {useEffect, useRef, useState} from "react";
import { select as d3_select, zoom as d3_zoom, zoomIdentity as d3_zoomIdentity } from "d3";
import { forceSimulation as d3_forceSimulation, forceLink as d3_forceLink, forceManyBody as d3_forceManyBody,
    forceCenter as d3_forceCenter} from "d3-force";
import { drag as d3_drag } from "d3-drag";
import { read as dotRead } from "graphlib-dot"
import {callAPI, out} from "./utils"

export default function FSM({_agentName_, _playPauseStatus_}) {
    out("[FSM] _agentName_: " + _agentName_ + ", _playPauseStatus_: " + _playPauseStatus_);

    // reference to the SVG area where the graph will de displayed
    const svgRef = useRef();

    // references to the node or edge that are currently highlighted (or zoomed in)
    const highlightedNodeRef = useRef(null);
    const highlightedEdgeRef = useRef(null);

    // node or ende that must highlighted (once they are set, there is usually an animation running)
    const [node2Highlight, setNode2Highlight] = useState(null);
    const [edge2Highlight, setEdge2Highlight] = useState(null);

    // state of the SVG area (structured as {width: a, height: b})
    const [svgSize, setSvgSize] = useState({width: 0, height: 0});

    // flag to recall when the data is being loaded through the API call
    const [loading, setLoading] = useState(true);

    // flag to signal that drawing operation has ended
    const [doneDrawing, setDrawingDone] = useState(false);

    // data in GraphViz format (string)
    const [graphvizDotStringData, setGraphvizDotStringData] = useState(null);

    // the D3 simulation engine
    const [simulation, setSimulation] = useState(null);

    // fetch the GraphViz string of the FSM
    useEffect(() => {
        out("[FSM] useEffect *** fetching data (get behaviour, agent name: " + _agentName_ + ") ***");
        setLoading(true); // loading started

        callAPI('/get_behav', "agent_name=" + _agentName_,
            (x) => {
                setGraphvizDotStringData(x);
                setLoading(false);  // done loading
            },
            () => setGraphvizDotStringData(null),  // clearing
            () => {
            })
    }, []);

    // fetch the current state/action
    useEffect(() => {
        if (doneDrawing && graphvizDotStringData && svgSize) {
            out("[FSM] useEffect *** fetching data (get behaviour status, agent name: " + _agentName_ + ") ***");

            callAPI('/get_behav_status', "agent_name=" + _agentName_,
                (x) => {
                    if (x.state != null && x.action == null) {
                        setNode2Highlight("node" + x.state); // state ID (manually defined state ID in GraphViz)
                    } else if (x.action != null && x.state == null) {
                        setEdge2Highlight("edge" + x.action);  // edge ID (manually defined edge ID in GraphViz)
                    } else {
                        throw new Error("Unknown behaviour status (state: " + x.state + ", action: " + x.action + ")");
                    }
                },
                () => {
                },
                () => {
                }
            );
        } else {
            out("[FSM] useEffect *** fetching data (get behaviour status, skipping - too early) ***");
        }
    }, [doneDrawing, graphvizDotStringData, svgSize, _playPauseStatus_]);

    // resize the FSM drawing
    useEffect(() => {
        out("[FSM] useEffect *** resizing ***");

        const handleResize = () => {
            if (svgRef.current) {
                const {width, height} = svgRef.current.getBoundingClientRect();
                out("[FSM] handleResize -> setSvgSize to width: " + width + ", height: " + height);
                setSvgSize({width, height});
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
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);  // clearing event
            resizeObserver.disconnect(); // clean up ResizeObserver
        };
    }, [graphvizDotStringData]);

    // highlight node or edge (triggered by changes to "node2Highlight" or to "edge2Highlight")
    // notice that "node2Highlight" and "edge2Highlight" are the IDs manually associated to the GraphViz nodes and edges
    useEffect(() => {
        out("[FSM] useEffect *** highlighting node or edge ***");

        // node
        if (node2Highlight && doneDrawing && svgSize.width !== 0 && svgSize.height !== 0) {
            setNode2Highlight(null);
            setEdge2Highlight(null);

            out("[FSM] Highlighting (clicking on) node '" + node2Highlight + "'");
            const targetNode = d3_select(svgRef.current).select("#" + node2Highlight);

            if (!targetNode.empty()) {
                targetNode.dispatch("click"); // dispatching a "click" event on node
            } else {
                out("[FSM] Node with id '" + node2Highlight + "' not found");
            }
        }

        // edge
        if (edge2Highlight && doneDrawing && svgSize.width !== 0 && svgSize.height !== 0) {
            setNode2Highlight(null);
            setEdge2Highlight(null);

            out("[FSM] Highlighting edge '" + edge2Highlight + "'");
            const targetEdge = d3_select(svgRef.current).select("#" + edge2Highlight);

            if (!targetEdge.empty()) {
                targetEdge.dispatch("click"); // dispatching a "click" event on edge
            } else {
                out("[FSM] Edge with id '" + edge2Highlight + "' not found");
            }
        }
    }, [doneDrawing, svgSize, node2Highlight, edge2Highlight]); // when drawing is done,
                                                                      // when size changes,
                                                                      // when behaviour status is loaded

    // drawing operations (when the data is fully loaded and when the size changes)
    useEffect(() => {
        if (!graphvizDotStringData || svgSize.width === 0 || svgSize.height === 0) {
            out("[FSM] useEffect *** drawing *** (data or svgSize not ready yet, skipping)");
            return;
        }

        const {width, height} = svgSize;
        out("[FSM] useEffect *** drawing *** (data ready, width: " + width + ", height: " + height + ")");

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
            const scale = 3;
            svg.transition().duration(750)
                .call(zoom.transform, d3_zoomIdentity.translate(width / 2, height / 2)
                    .scale(scale).translate(-centerX, -centerY));
        };

        const zoomToNode = (node) => {
            const scale = 2;
            svg.transition().duration(750)
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
        const svg = d3_select(svgRef.current);

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
                id: attributes.id
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

        // stopping simulations that might be running due to previous calls
        if (simulation) {
            out("[FSM] stopping previous simulation");
            simulation.stop();
        }

        // creating a new simulation (and stopping it, immediately)
        const sim = d3_forceSimulation(graphData.nodes)
            .force("link", d3_forceLink(graphData.links).id((d) => d.id).distance(100))
            .force("charge", d3_forceManyBody().strength(-200))
            .force("center", d3_forceCenter(width / 2, height / 2))
            .stop();

        // saving the state with the new simulation
        setSimulation(sim);

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
                highlightEdge(d);
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
            .attr("x", (d, i) => (d.source.x + d.target.x) / 2 + (i * 30))
            .attr("y", (d, i) => (d.source.y + d.target.y) / 2 + (i * 30))
            .attr("id", (d) => (d.id))
            .style("cursor", "default")
            .text((d) => d.label)
            .on("click", (event, d) => {
                event.stopPropagation();
                highlightEdge(d);
            });

          // dynamically update the width and height of the node based on the label's bounding box (used many times)
        function updateNodePositionsAndSizes() {
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
        }

        // drag action, applied to nodes (will be provided as attribute in what follows)
        const drag = d3_drag()
            .on("start", (event, d) => {

                // fixing position of nodes and labels (to synch them at the beginning)
                d.fx = d.x;
                d.fy = d.y;
                label.filter((ld) => ld.id === d.id)
                    .attr("x", d.x)
                    .attr("y", d.y);
            })
            .on("drag", (event, d) => {

                // updating node position in the event data structure (not on the node yet)
                d.fx = event.x;
                d.fy = event.y;
                d.x = event.x;
                d.y = event.y;

                // running a simulation tick to let the other nodes/edges move in function of the current movement
                sim.tick();

                // update node position using the event data structure (now on the node)
                node.attr("x", d => d.x - d.width / 2)
                    .attr("y", d => d.y - d.height / 2);

                // update edge positions during the drag
                link.attr("d", getLinkPath);

                // update edge labels' positions
                edgeLabel.attr("x", (d) => (d.source.x + d.target.x) / 2)
                    .attr("y", (d) => (d.source.y + d.target.y) / 2);

                // update node labels' positions to match the node's center
                label.attr("x", d => d.x)
                    .attr("y", d => d.y);

                // dynamically update the width and height of the node based on the label's bounding box
                updateNodePositionsAndSizes();
            })
            .on("end", (event, d) => {

                // when drag ends, we do the same thing done above
                updateNodePositionsAndSizes();

                // update label position on drag end to align with the node
                label.filter((ld) => ld.id === d.id)
                    .attr("x", d.x)
                    .attr("y", d.y);

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
            .text((d) => d.name)
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("font-size", "14px")
            .style("pointer-events", "none");

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

        out("[FSM] Graph simulation/drawing ended");
        setDrawingDone(true);  // marking the drawing has completed
    }, [graphvizDotStringData, svgSize]); // redraw when the data is loaded and when the size changes

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
