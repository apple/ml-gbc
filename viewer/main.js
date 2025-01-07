import Showdown from 'showdown';

import './style.css'

import { options } from './vis-settings.js'
import { parseName, printVertex, classname, nodeJumper, bbox } from './utils.js';
import config from './config.js';


//region constants
const SIZEMAP = {
  "image": 20,
  "global": 20,
  "relation": 15,
  "composition": 15,
  "entity": 10,
}
//endregion

//region Setup
var htmlToMd = new Showdown.Converter();
var nodes, edges, img_url;
var searchQuery = undefined;
//endregion

//region Search
const searchInput = document.getElementById("searchInput");
var endinputTimeout = undefined;
searchInput.addEventListener("input", (event) => {
  if(endinputTimeout) clearTimeout(endinputTimeout);
  endinputTimeout = setTimeout(() => {
    searchQuery = new URLSearchParams({ target: event.target.value });
    updateIndex();
  }, 500);
});
//endregion


//region Data selection
var graphs = [];
var totalPages = Math.ceil(graphs.length / 10);
let currentPage = 0;
function loadData(currentGraph) {
  // console.log("currentGraph: " + JSON.stringify(currentGraph))
  if (currentGraph.img_url && currentGraph.img_url.trim() !== '') {
    img_url = currentGraph.img_url;
    console.log("img_url: " + img_url);
  } else {
    // Throw an error if both img_path and mg_url are empty
    console.error("img_url is empty");
    img_url = null;
  }
  let history = [];
  let new_nodes = [];
  let new_edges = [];
  // Convert vertices from dict to array
  if (currentGraph.vertices && typeof currentGraph.vertices === 'object' && !Array.isArray(currentGraph.vertices)) {
    currentGraph.vertices = Object.values(currentGraph.vertices);
  }
  for (let vertex of currentGraph.vertices) {
    let desc = "";
    for (let descs of vertex.descs) {
      desc += descs.text;
      desc += "\n\n\n";
    }
    for (let edge of vertex.in_edges) {
      edge = {
        "arrows": "to",
        "width": 1,
        "from": edge.source,
        "to": edge.target,
      }
      if (!history.includes(`${edge.from}-${edge.to}`)) {
        new_edges.push(edge)
        history.push(`${edge.from}-${edge.to}`);
      }
    }
    new_nodes.push({
      "id": vertex.vertex_id,
      "bbox": vertex.bbox,
      "label": vertex.vertex_id ? parseName(vertex.vertex_id) : vertex.label,
      "group": vertex.label,
      "shape": "dot",
      "size": SIZEMAP[vertex.label],
      "data": vertex
    })
  }
  nodes = new vis.DataSet(new_nodes);
  edges = new vis.DataSet(new_edges);
  nodes.forEach((node) => {
    node.content = printVertex(node.data, nodes, edges, options);
  });
  setupNetwork();
}
function loadDataFetch(id) {
  fetch(`${config.api_url}/graph/${id}`, {
    method: 'GET',
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data)
      loadData(data);
    })
}

const sideBar = document.getElementById("nav");
function updateSidebar(page, partialDatas) {
  document.getElementById("page").innerText = `${page + 1}/${totalPages}`;
  if (page == 0) {
    document.getElementById("prevBtn").disabled = true;
    document.getElementById("nextBtn").disabled = false;
  } else if (page == totalPages - 1) {
    document.getElementById("prevBtn").disabled = false;
    document.getElementById("nextBtn").disabled = true;
  } else {
    document.getElementById("prevBtn").disabled = false;
    document.getElementById("nextBtn").disabled = false;
  }

  // let partialDatas = graphs.slice(page * 10, (page + 1) * 10);
  let html = "";
  partialDatas.forEach((data, index) => {
    html += `
    <button class="item-nav btn btn-outline-info w-100 p-2 mt-2 ${index==0 ? "active" : ""}" id="data${data[0]}">
      <li class="nav-item">#${data[0] + 1} - ${data[1].short_caption}</li>
    </button>
`
  })
  sideBar.innerHTML = html;
  currentPage = page;
  partialDatas.forEach((data) => {
    data.push(document.getElementById(`data${data[0]}`));
    document.getElementById(`data${data[0]}`).addEventListener('click', () => {
      loadDataFetch(data[0]);
      data[2].classList.add("active");
      document.querySelector(".item-nav.active").classList.remove("active");
    })
  })
}
function updatePage(page) {
  fetch(`${config.api_url}/index/${page}?${searchQuery ?new URLSearchParams(searchQuery).toString(): ""}`, {
    method: 'GET',
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data)
      updateSidebar(page, data);
      loadDataFetch(data[0][0]);
    })
}
function updateIndex(){
  // put searchQuery into data when exist
  fetch(`${config.api_url}/index?${searchQuery ?new URLSearchParams(searchQuery).toString(): ""}`, {
    method: 'GET'
  })
  .then((response) => response.json())
  .then((data) => {
    console.log(data)
    totalPages = data;
    updatePage(0);
  })
}
document.getElementById("prevBtn").addEventListener('click', () => {
  updatePage(currentPage - 1);
})
document.getElementById("nextBtn").addEventListener('click', () => {
  updatePage(currentPage + 1);
})
//endregion


//region Vis.Network
var container = document.getElementById("mynetwork");
var bbox_wrapper = document.getElementById("bbox-wrapper");
// Setup Network
function setupNetwork() {
  var data = {
    nodes: nodes,
    edges: edges,
  };
  document.getElementById("img").src = "";
  if(img_url != "none"){
    document.getElementById("urlError").classList.remove("d-flex");
    document.getElementById("urlError").classList.add("d-none");
    document.getElementById("img").src = img_url;
  }else{
    document.getElementById("urlError").classList.remove("d-none");
    document.getElementById("urlError").classList.add("d-flex");
  }
  var network = new vis.Network(container, data, options);
  let bboxInnerHtmlCache = "";
  let currentMode = "vertex";
  function handleSelect(data) {
    console.log(data)
    if (data.nodes.length == 1) {
      currentMode = "vertex"
      bbox_wrapper.innerHTML = bbox(nodes.get(data.nodes[0]), 0.2)
      bboxInnerHtmlCache = bbox_wrapper.innerHTML
      let selected = data.nodes[0];
      document.querySelector("#captionTitle").innerHTML = "Vertex Information";
      document.querySelector("#captionViewer").innerHTML = htmlToMd.makeHtml(nodes.get(selected).content);
    } else if (data.edges.length == 1) {
      currentMode = "edge"
      bbox_wrapper.innerHTML = ""
      let edge = edges.get(data.edges[0]);
      let nodeFrom, nodeTo;
      nodeFrom = nodes.get(edge.from, 0.2);
      nodeTo = nodes.get(edge.to, 0.15, 0.25);
      bbox_wrapper.innerHTML += bbox(nodeFrom);
      bbox_wrapper.innerHTML += bbox(nodeTo);
      document.querySelector("#captionTitle").innerHTML = "Edge Information";
      document.querySelector("#captionViewer").innerHTML = `From : ${nodeJumper(nodes, nodeFrom.id)}<br>To : ${nodeJumper(nodes, nodeTo.id)}`;
      bboxInnerHtmlCache = bbox_wrapper.innerHTML
    } else {
      return;
    }
    
    setTimeout(() => {
      setupJumper();
    }, 5);
  }
  function mouseOverBbox(node){
    if(currentMode === "vertex"){
      bbox_wrapper.innerHTML = bboxInnerHtmlCache
      bbox_wrapper.innerHTML += bbox(node, 0.3, 0, 0.5)
    }else{
      bbox_wrapper.innerHTML = bbox(node, 0.3, 0, 0.5)
    }
  }
  function mouseOutBbox(node){
    bbox_wrapper.innerHTML = bboxInnerHtmlCache
  }
  function setupJumper() {
    nodes.forEach((node) => {
      let nodeJumpers = document.querySelectorAll(`.to-${classname(node.id).replace(/ /g, "_")}`);
      for (let nodeJumper of nodeJumpers) {
        nodeJumper?.addEventListener('click', () => {
          network.selectNodes([node.id], true);
          handleSelect({ nodes: [node.id], edges: network.getConnectedEdges([node.id]) });
        })
        nodeJumper.addEventListener('mouseover', (event) => {
          console.log(node.id, "mouseover", event)
          mouseOverBbox(node)
        })
        nodeJumper.addEventListener('mouseout', (event) => {
          console.log(node.id, "mouseout", event)
          mouseOutBbox(node)
        })
      }
    })
  }
  bbox_wrapper.innerHTML = "";
  document.querySelector("#captionTitle").innerHTML = "Click Vertex or Edge on the Graph to view its information";
  document.querySelector("#captionViewer").innerHTML = "";
  network.on("click", handleSelect);
  network.selectNodes([""], true);
  handleSelect({
    nodes: network.getSelectedNodes(),
    edges: network.getSelectedEdges(),
  })
}
//endregion


updateIndex()
