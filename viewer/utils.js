function parseName(label) {
    if(label==''){
        return 'empty(root)';
    }
    return label.replace(/_/g, ' ').replace(/\|/g, ' / ').replace(/[^a-zA-Z0-9_/ ]/g, '');
}

// function classname(string) {
//     return string.replace(/[.*+?^${}()|[\]\\]/g, '_-_');
// }
function classname(str) {
    return str.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function nodeJumper(nodes, id, text) {
    let node = nodes.get(id);
    text = text == undefined ? node.label : text;
    return `<a class="to-${classname(id.replace(/ /g, '_'))} node node-${node.group}">${text}</a>`
}

function rewriteCaption(description, texts, nodes) {
    var history = [""];
    texts = texts.filter((value, index) => {
        let result = !history.includes(value[0]);
        history.push(value[0]);
        return result
    });
    console.log("texts", texts)
    for (let [text, id] of texts) {
        // Use 'i' so that it's case-insensitive
        var re = new RegExp(
            `([^a-zA-Z])(${text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})([^a-zA-Z])`,
            "gi"
        );
        // Create the regular expression with case insensitivity and word boundary handling
        var re = new RegExp(
            `(^|[^a-zA-Z])(${text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})([^a-zA-Z]|$)`,
            "gi"
        );
        description = description.replace(re, (match, p1, p2, p3) => {
            // console.log(`Original text: ${text}`);
            // console.log(`Found in description: ${p2}`);
            return `${p1}${nodeJumper(nodes, id, p2)}${p3}`;
        });
    }
    return description
}

function printVertex(vertex, nodes, edges, options) {
    let result = "";
    result += "**Vertex ID: " + parseName(vertex.vertex_id) + "‌**\n";
    result += "\n**Label: " + vertex.label + "**\n ---\n";

    if (vertex.in_edges.length > 0) {
        result += "\nIn Edges: \n";
        for (let in_edge of vertex.in_edges) {
            result += `- ${in_edge.text} (${nodeJumper(nodes, in_edge.source)} -> this)\n`;
        }
        result += "\n<br>";
    }

    let out_edge_texts = [];
    if (vertex.out_edges.length > 0) {
        result += "\nOut Edges: \n";
        for (let out_edge of vertex.out_edges) {
            result += `- ${out_edge.text}  (this -> ${nodeJumper(nodes, out_edge.target)})\n`;
            out_edge_texts.push([out_edge.text, out_edge.target]);
        }
        result += "\n<br>";
    }

    result += "\nCaptions:\n";
    for (let { text, label } of vertex.descs) {
        text = rewriteCaption(text, out_edge_texts, nodes);
        result += `- (**${label}**) ${text}\n`;
    }
    return result
}

function bbox(node, width, offset, transparency) {
    var style = "";
    offset = offset || 0;
    style += `top:${node.bbox.top * 100 + offset}%;`;
    style += `left:${node.bbox.left * 100 + offset}%;`;
    style += `bottom:${(1 - node.bbox.bottom) * 100 + offset}%;`;
    style += `right:${(1 - node.bbox.right) * 100 + offset}%;`;
    style += `border: ${width ? width : '.2'}rem solid var(--${node.group}-color);`;
    style += `opacity: ${transparency ? transparency : 1};`;
    return `<div class="bbox" style="${style}"></div>`;
}

export { parseName, printVertex, classname, nodeJumper, bbox }
