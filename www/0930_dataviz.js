Shiny.addCustomMessageHandler("testver",
  function(message) {
    var graph = message.g;
    var bib = message.b;
    var author = message.a;
    var updatedArray = message.u;
    var isChanged = message.c;
    
    var useGroupInABox = true,
  drawTemplate = true,
  template = "force";

d3.select("#checkGroupInABox").property("checked", useGroupInABox);
d3.select("#checkShowTreemap").property("checked", drawTemplate);
d3.select("#selectTemplate").property("value", template);
var color = d3.scaleOrdinal(d3.schemeCategory10);

// var width = 1500,
//     height = 1200;
// var margin = 100;

// var svg = d3.select("#chart").append("svg")
//     .attr("width", width)
//     .attr("height", height);
// var authorSvg = d3.select("#author").append("svg")
//     .attr("width", width)
//     .attr("height", height);
// var chart_right = authorSvg.append("g")
//     .attr("transform", `translate(${[margin/2,margin/2]})`);

// var chart_left = svg.append("g")
//     .attr("transform", `translate(${[margin/2,margin/2]})`)
//     // chart_right.selectAll("g").remove()
//     // chart_right = authorSvg.append("g")
//     //     .attr("transform", `translate(${[margin/2,margin/2]})`);
// var innerChart = chart_left.attr('class','innerChart');

// var innerChart = chart_left.attr('class','innerChart');
//updated array passed from Shiny
// var updatedArray = []

var pubArray = [];

//nodes: the svg container for the keywords graph.
//keywordsArray: updated array of selected keywords by users from Shiny

var keywordsArray = new Set(updatedArray);
//parameter: publication file, keywords array
//returns the populated array with publications that correspond to keywords.
function kwfilter(bib, keywordsArray){
  //bib: publication object
//keywordsArray: array of selected keywords by users
  // var pubArray = [];
    
    keywordsArray.forEach(function(keyword){
        
        
        bib.forEach(function(b){
          if (b.keywords.includes(keyword)){
            //When R reads a json file, it converts array with one author to a string
            //Thus the conditional accounts for when that happens and correctly populates the array.
            if(typeof(b.bib_author) === 'string'){
              pubArray.push(b.bib_author);
              }else{
                b.bib_author.forEach(function(a){
                pubArray.push(a);
            });
          }
      }
      });
            
    });
        
  return pubArray;
    
}//end of kwfilter

function showText(text){
  if(text.classed("hidden") == true){
    text.classed("hidden", false);
    text.attr("display", "none");
  }else{
    text.classed("hidden", true);
    text.attr("display", "block");
  }
}
function drawNetwork(bib, keywordsArray, author){
  var width = 1500,
        height = 1200;
    var margin = 100;
  var authorSvg = d3.select("#author").append("svg")
        .attr("width", width)
        .attr("height", height);

  var chart_right = authorSvg.append("g")
        .attr("transform", `translate(${[margin/2,margin/2]})`);
  var nodeId = new Set();
  var edgeList = chart_right.append("g").attr("transform", `translate(${[width/2 + margin/2, margin/2]})`);
  edgeList.append("text");
    //we use the filter function to get authors
    //var authorArray = kwfilter(bib, keywordsArray);
    // var nodes = author.nodes.filter(function(node) { return authorArray.includes(node.keywords);} )
     // console.log(authorArray);
      //nodes are filteredAuthors
      //we filter the authors based on their co-authorship
     // const brush = d3.brush();
        // brush.filter(event => !event.ctrlKey
        // && !event.button
        // && (event.metaKey
        // || event.target.__data__.type !== "overlay"))
     //   brush.on("end", detail);
    //chart_right.append("g")
    //    .attr("class", "brush")
      var nodes = author.nodes
      
      nodes.forEach(n => {
        nodeId.add(n.id);
      })
      //console.log(nodeId);
      //console.log(typeof(author.links[0].source))
      //console.log(nodeId.has(57196081184));
      //author.links.forEach(l => {console.log(l.source);});
      lk = author.links
      .filter(l => {return (nodeId.has(l.source) && nodeId.has(l.target))});
      //console.log("link: "+ lk);
      //   //console.log(author.links.filter(l => l.source.id == 27));
      //   var plz = author.links.filter(l => nodeId.forEach(function(d){
      //     return l.source == d || l.target == d;
      //   }));
      //   console.log(plz);
      var degreeSize = d3.scaleLinear()
          .domain([d3.min(nodes, function(d) {return d.num_coauthors; }),
            d3.max(nodes, function(d) {return d.num_coauthors; })])
          .range([3,25]);
      var nodeOpcacity = d3.scaleLinear()
          .domain([d3.min(nodes, function(d) {return d.crys_score; }),
            d3.max(nodes, function(d) {return d.crys_score; })])
          .range([0.2, 1.0]);

      
      
      var groupingForce = forceInABox()
        .strength(0.02) // Strength to foci
        .template(template) // Either treemap or force
        .groupBy("group_id") // Node attribute to group
        .links(lk) // The graph links. Must be called after setting the grouping attribute
        .enableGrouping(useGroupInABox)
        .nodeSize(5)
        .linkStrengthIntraCluster(0.05)
        .size([width, height]) // Size of the chart
      //chart_right.selectAll('g').remove();
      // const sim = d3.forceSimulation(nodes)
      // .force("charge", d3.forceManyBody())
      // .force("x", d3.forceX(width/2).strength(0.03))
      // .force("y", d3.forceY(height/2).strength(0.03));
    const sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(lk).id(d => d.id).distance(1000).strength(groupingForce.getLinkStrength))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(width/4 - margin/2, height/ 2 - margin/2))
        .force("group_id", groupingForce);


   // function detail({selection}){
    
      //    if (selection) { 
        
       //     const [[x0, y0], [x1, y1]] = selection;
        
       //     var nodeFiltered = node.filter(d => x0 <= d.x && d.x < x1 && y0 <= d.y && d.y < y1 || x0 <= (d.x + d.w) && d.x < x1 && y0 <= (d.y + d.h) && d.y < y1);
       //     var textFiltered = text.filter(d => x0 <= d.x && d.x < x1 && y0 <= d.y && d.y < y1 || x0 <= (d.x + d.w) && d.x < x1 && y0 <= (d.y + d.h) && d.y < y1);
        
        
         //   nodeFiltered.each(function(d){
          //console.log("selected: "+ d3.select(this).classed("selected"));
          //    if (d3.select(this).classed("selected")){
            
          //      nodeFiltered.style("fill", function(d){return(color(d.group_id))})
            //      text
             //       .style("fill", "black")
           //         .style("stroke", "black");
          //      } else {
          //        keywordsArray.add(d.keywords);
            //      nodeFiltered.style("fill", "red")
            //      textFiltered
            //      .style("fill", "red")
           //       .style("stroke", "red");
           //     }  
        //    })
       //     nodeFiltered.classed("selected", !nodeFiltered.classed("selected"));
        
        
        
        // var filteredData = nodeFiltered.data();
        // filteredData.forEach(function(d) { keywordsArray.add(d.keywords); });
        
      // keywordsArray.forEach(function(d) {
      //   console.log(d)})
  
      
  
  
      
      
      //  }
// }
  

  var link_d = chart_right
      .append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(lk)
      .join("line")
      .attr("stroke-width", 1);
      

  var node_s = chart_right
      .append("g")
      .selectAll("circle")
      .data(author.nodes)
      .join("circle")
      .attr("r", function(d) { return degreeSize(d.ScholarlyOutput); })
      .classed("selected", true)
        .attr("fill", d => {return color(d.group_id)})
        .attr("opacity", d => {return nodeOpcacity(d.crys_score)})
        .attr('id', (d, i) => {return "circle_" + i})
      .call(drag(sim));
  
  var text_s = chart_right
    .append("g")
    .selectAll("text")
    .data(author.nodes)
        .join("text")
        .attr("dx", 15)
        .attr("dy", ".35em")
        .attr('id', (d, i) => {return "text_" + i})
        .style("stroke-width", "0.6")
        .style("stroke", "black")
        .text(function(d) { return d.author })
        .classed("hidden", false);

        node_s.on("mouseenter", (evt, d) => {
    
      
          lk = link_d.filter(l => l.source === d.id || l.target === d.id);
          
             lk.each(function(d,i){
            // console.log(d)
            edgeList.select("text")
            .attr("x", 15)
            .attr("y", 50)
            .text("Source: ")
            .style("stroke", "red")
            .append('tspan')
            .text(d.source_keyword)
            .style("stroke-width", "0.7")
            .style("stroke", "yellow")
            .append('tspan')
            
            .attr("x", 15)
            .attr("y", 70)
            .text("Target: ")
            .style("stroke", "red")
            .append('tspan')
            .text(d.target_keyword)
            .style("stroke-width", "0.7")
            .style("stroke", "steelblue");
             })
          var selectedID = d3.select(event.target).attr("id").split("_")[1];
          var textID = "#text_" + selectedID;
          
          text_s
              .style("fill-opacity", "0.1")
              .style("stroke-opacity", "0.1");
          d3.select(textID)
            .style('stroke', "yellow")
            .style("stroke-width", "0.7")
            .style('stroke-opacity', "0.9");
      
          node_s.style("fill-opacity", "0.2");
      
            
          d3.select(evt.target)
                .style('fill-opacity', "0.9")
                .style('fill', "orange");
      
          // nodeFiltered = node.filter(n => (n.group.some(element => {
          //   return d.group.includes(element);})));
          // textFiltered = text.filter(t => (t.group.some(element => {
          //   return d.group.includes(element);})));
        
          // textFiltered
          // .style('stroke', 'orange')
          // .style("stroke-opacity", "1");
          // nodeFiltered
          // .style('fill', 'orange')
          // .style("fill-opacity", "0.6");
      
        //   node.style('fill', function (n) { return n.group.some(element => {
        // return d.group.includes(element);}) ? 'orange' : 'blue';})
          
        //     link
        //       .style('stroke', function (link_d) { return link_d.group.some(element => {
        // return d.group.includes(element);}) ? '#69b3b2' : '#b8b8b8';})
        //       .style('stroke-opacity', function (link_d) { return link_d.group.some(element => {
        // return d.group.includes(element);}) ? '0.9' : '0.2';})
        //       .style('stroke-width', function (link_d) { return d.isTriple && link_d.group.some(element => {
        // return d.group.includes(element);}) ? 4 : 1;});
              
            link_d
              .filter(l => (l.source.id === d.id || l.target.id === d.id))
              .style('stroke', "red")
              .style("stroke-opacity", "0.9")
              .style("stroke-width", 4)
              
              })
          .on("mouseleave", (evt, d) => {
            var selectedID = d3.select(event.target).attr("id").split("_")[1];
          var textID = "#text_" + selectedID;
            d3.select(textID)
              .style("stroke-width", "0.7")
              .style("stroke", "black")
              .style('stroke-opacity', "0.7");
            d3.select(evt.target)
              .style('fill-opacity', "0.7")
              .style("fill", function(d) { return color(d.group_id); });
            node_s
              .style("fill-opacity", "0.7")
              .style("fill", function(d) { return color(d.group_id);});
            text_s
              .style("stroke", "black")
              .style("fill-opacity", "0.7")
              .style("stroke-opacity", "0.7");
      
            link_d
                .style('stroke', "#999")
                .style("stroke-width", d => d.weight)
            
            
          });
          node_s.on("click", (evt, d) => {
            link_d
              .filter(l => (l.source.id === d.id || l.target.id === d.id))
              .style('stroke', "red")
              .style("stroke-opacity", "0.9")
              .style("stroke-width", 4)
              
              })
          
    function drag(sim) {    
        function dragstarted(event, d) {
          if (!event.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }
    
        function dragged(event, d) {
          d.fx = event.x;
          d.fy = event.y;
        }
    
        function dragended(event, d) {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        } 
        return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    }
        sim.on("tick", function() {
            link_d.attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
        node_s.attr("cx", d => d.x)
                .attr("cy", d => d.y);
            //node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
            text_s
              .attr("x", d => d.x)
              .attr("y", d => d.y);
          });
      
      d3.select("#textToggle").on("click", function() {
          showText(text_s);
      })
  }
  
  function highlightKwarray(nodes, keywordsArray){
  nodes.style("fill", d=>{return color(d.isTriple);});
    if(typeof(keywordsArray) === 'string'){
        nodes.filter(function(d){
            return d.keywords == keywordsArray;
            //highlight all nodes in the updated array
        }).style("fill", "red");
      //  console.log("here is string?")
    }
    else {
    keywordsArray.forEach(function(keyword){
        nodes.filter(function(d){
            return d.keywords == keyword;
            //highlight all nodes in the updated array
        }).style("fill", "red");
        });
    }
}//end of highlightKwarray

  function updateNodes(keywordArray){
    var nodes = d3.select("#chart").selectAll(".nodes");
    //console.log(nodes);
    highlightKwarray(nodes, keywordArray);
  }
  // r2d3.onRender(function(graph, svg, width, height, options) {

//user needs to press command key to brush
  function magnify(array){
    array = Array.from(array);
    var width = 700,
        height = 500;
    var margin = 50;
    // var authorSvg = d3.select("#author").append("svg")
    //     .attr("width", width)
    //     .attr("height", height);
    var chart = d3.select("#zoomView").append("svg")
        .attr("width", width)
        .attr("height", height);
    var zoomView = chart.append("g")
        .attr("transform", `translate(${[margin/2,margin/2]})`);
    zoomView.attr('class','zoomView');
    
    nodes = graph.nodes.filter(function(node) { return array.includes(node.keywords);} )
    //console.log("the nodes" + nodes);
    var nodeId = new Set();
    nodes.forEach(n => {
        nodeId.add(n.id);
      })
    
    lk = graph.links
      .filter(l => {return (nodeId.has(l.source.id) && nodeId.has(l.target.id))});
    
    var degreeSize = d3.scaleSqrt()
        .domain([d3.min(nodes, function(d) {return d.degree; }),
          d3.max(nodes, function(d) {return d.degree; })])
        .range([5,10]);
  var groupingForce = forceInABox()
            .strength(0.05) // Strength to foci
            .template(template) // Either treemap or force
            .groupBy("isTriple") // Node attribute to group
            .links(lk) // The graph links. Must be called after setting the grouping attribute
            .enableGrouping(useGroupInABox)
            .nodeSize(5)
            .linkStrengthIntraCluster(0.05)
            .size([width, height]) // Size of the chart
  // var simulation = d3
  //     .forceSimulation(graph.nodes)
  //     .force("link", d3.forceLink(graph.links).id(d => d.id))
  //     .force("charge", d3.forceManyBody())
  //     .force("center", d3.forceCenter(width / 2, height / 2));
  var simulation = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(lk)
      .id(d => d.id)
      .distance(200)
      .strength(groupingForce.getLinkStrength))
    .force("charge", d3.forceManyBody())
    //.force("center", d3.forceCenter(width / 2, height / 2));
    .force("x", d3.forceX(width/4).strength(0.05))
    .force("y", d3.forceY(height/2).strength(0.05));

  
 //simulation.force("isTriple", groupingForce);

var link = zoomView
    .append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(lk)
    .join("line")
    .attr("stroke-width", d => d.weight);


var node = zoomView
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", function(d) { return degreeSize(d.degree); })
    .attr("fill", "steelblue" )
    .attr('id', (d, i) => {return "circle_" + i})
    .attr("selected", false)
    .classed("selected", true)
    .call(drag(simulation));
  //node.attr('class', 'nodes')
var text = zoomView
  .append("g")
  .selectAll("text")
  .data(nodes)
      .join("text")
      .attr("dx", 15)
      .attr("dy", ".35em")
      .attr('id', (d, i) => {return "text_" + i})
      .style("stroke-width", "0.7")
      .style("stroke", "black")
      .style("fill", "black")
      .text(function(d) { return d.keywords });

    //text.attr('class', 'texts')
    function drag(simulation) {    
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
  
      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }
  
      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      } 
      return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  }
  simulation.on("tick", function() {
    
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });
  node.attr("cx", d => d.x)
      .attr("cy", d => d.y);
  text
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

  }
  function draw(){
    var width = 1500,
        height = 1200;
    var margin = 100;

    var svg = d3.select("#chart").append("svg")
        .attr("width", width)
        .attr("height", height);
    // var authorSvg = d3.select("#author").append("svg")
    //     .attr("width", width)
    //     .attr("height", height);
    // var chart_right = authorSvg.append("g")
    //     .attr("transform", `translate(${[margin/2,margin/2]})`);

    var chart_left = svg.append("g")
        .attr("transform", `translate(${[margin/2,margin/2]})`)
        // chart_right.selectAll("g").remove()
        // chart_right = authorSvg.append("g")
        //     .attr("transform", `translate(${[margin/2,margin/2]})`);
    var innerChart = chart_left.attr('class','innerChart');
    svg.call(
      d3.zoom()
          .scaleExtent([.1, 4])
          .on("zoom", function(event) { chart_left.attr("transform", event.transform); })
  );
    const brush = d3.brush();
        // brush.filter(event => !event.ctrlKey
        // && !event.button
        // && (event.metaKey
        // || event.target.__data__.type !== "overlay"))
        brush.on("end", detail);
    innerChart.append("g")
        .attr("class", "brush")
    
      var label = {
          'nodes': [],
          'links': []
      };
      
      graph.nodes.forEach(function(d, i) {
          //label.nodes.push({node: d});
          label.nodes.push({node: d});
      });
      graph.links.forEach(function(d, i) {
        label.links.push({
          source: d.source,
          target: d.target
      });
  })
  //console.log(label.nodes);
  var labelSim = d3
    .forceSimulation(label.nodes)
    .force("link", d3.forceLink(label.links)
      //.id(d => d.id)
      .distance(0)
      .strength(2))
    .force("charge", d3.forceManyBody().strength(-50));
    
  var degreeSize = d3.scaleSqrt()
        .domain([d3.min(graph.nodes, function(d) {return d.weight; }),
          d3.max(graph.nodes, function(d) {return d.weight; })])
        .range([5,15]);
  var groupingForce = forceInABox()
            .strength(-100) // Strength to foci
            .template(template) // Either treemap or force
            .groupBy("weight") // Node attribute to group
            .links(graph.links) // The graph links. Must be called after setting the grouping attribute
            .enableGrouping(useGroupInABox)
            .nodeSize(5)
            .linkStrengthIntraCluster(0.01)
            .size([width, height]) // Size of the chart
  // var simulation = d3
  //     .forceSimulation(graph.nodes)
  //     .force("link", d3.forceLink(graph.links).id(d => d.id).distance(20)
  //     .strength(groupingForce.getLinkStrength))
  //     .force("charge", d3.forceManyBody())
  //     .force("center", d3.forceCenter(width / 2, height / 2));
  
  // var simulation = d3
  //   .forceSimulation(graph.nodes)
  //   .force("link", d3.forceLink(graph.links)
  //     .id(d => d.id)
  //     .distance(20)
  //     .strength(groupingForce.getLinkStrength))
  //   .force("charge", d3.forceManyBody().strength(-200))
  //   //.force("center", d3.forceCenter(width / 2, height / 2));
  //   .force("x", d3.forceX(width/4).strength(0.05))
  //   .force("y", d3.forceY(height/2).strength(0.05));
  var simulation = d3.forceSimulation(graph.nodes)
    .force("charge", d3.forceManyBody().strength(-3000))
    .force("center", d3.forceCenter(width / 2, height / 2))
    
    .force("x", d3.forceX(width / 2).strength(1))
    .force("y", d3.forceY(height / 2).strength(1))
    .force("link", d3.forceLink(graph.links).id(function(d) {return d.id; }).distance(500).strength(0.01))
  
 //simulation.force("isTriple", groupingForce);

var link = innerChart
    .append("g")
    .attr("stroke", "#999")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(graph.links)
    .join("line")
    .attr("stroke-width", d => d.weight);


var node = innerChart
    .append("g")
    .selectAll("circle")
    .data(graph.nodes)
    .join("circle")
    .attr("r", function(d) { return degreeSize(d.weight); })
    .attr("fill", "steelblue" )
    .attr('id', (d, i) => {return "circle_" + i})
    .attr("selected", false)
    .classed("selected", true)
    .call(drag(simulation));
  node.attr('class', 'nodes')

// var text = innerChart
//   .append("g")
//   .selectAll("text")
//   .data(graph.nodes)
//       .join("text")
//       // .attr("dx", 15)
//       // .attr("dy", ".35em")
//       .attr('id', (d, i) => {return "text_" + i})
//       .style("stroke-width", "0.7")
//       .style("stroke", "black")
//       .style("fill", "black")
//       .style("pointer-events", "none")
//       .text(function(d) { return d.keywords });

//     text.attr('class', 'texts')
var text = innerChart
  .append("g")
  .selectAll("text")
  .data(graph.nodes)
      .join("text")
      .attr("dx", 15)
      .attr("dy", ".35em")
      .attr('id', (d, i) => {return "text_" + i})
      .style("stroke-width", "0.7")
      .style("stroke", "black")
      .style("fill", "black")
      .style("pointer-events", "none")
      //.text("test");
      .text(function(d) { return d.keywords });

    text.attr('class', 'texts')
  
// var labelNode = innerChart.append("g").attr("class", "labelNodes")
//     .selectAll("text")
//     .data(label.nodes)
//     .join("text")
//     //.text(function(d, i) { return i % 2 == 0 ? "" : d.keywords; })
//     .text("test")
//     .style("fill", "#555")
//     .style("font-family", "Arial")
//     .style("font-size", 12)
//     .style("pointer-events", "none"); // to prevent mouseover/drag capture

  
  //const message = document.querySelector('.keywords');
        //const result = document.querySelector('#result');
    //    message.addEventListener('change', event => {
          // console.log(event.target.value);
      //    keywordsArray.add(event.target.value);
       //   console.log(keywordsArray);
      //    updateNodes(keywordsArray);
          //highlightKwarray(node, keywordsArray);
       // });

  function detail({selection}){
    
    if (selection) { 
      
      const [[x0, y0], [x1, y1]] = selection;
      
      var nodeFiltered = node.filter(d => x0 <= d.x && d.x < x1 && y0 <= d.y && d.y < y1 || x0 <= (d.x + d.w) && d.x < x1 && y0 <= (d.y + d.h) && d.y < y1);
      var textFiltered = text.filter(d => x0 <= d.x && d.x < x1 && y0 <= d.y && d.y < y1 || x0 <= (d.x + d.w) && d.x < x1 && y0 <= (d.y + d.h) && d.y < y1);
      
      
      nodeFiltered.each(function(d){
        //console.log("selected: "+ d3.select(this).classed("selected"));
        if (d3.select(this).classed("selected")){
          keywordsArray.delete(d.keywords);
          nodeFiltered.style("fill", function(d){return(color(d.isTriple))})
            text
              .style("fill", "black")
              .style("stroke", "black");
          } else {
            keywordsArray.add(d.keywords);
            nodeFiltered.style("fill", "red")
            textFiltered
            .style("fill", "red")
            .style("stroke", "red");
          }  
      })
      nodeFiltered.classed("selected", !nodeFiltered.classed("selected"));
      
      
      
      // var filteredData = nodeFiltered.data();
      // filteredData.forEach(function(d) { keywordsArray.add(d.keywords); });
      
    // keywordsArray.forEach(function(d) {
    //   console.log(d)})

    console.log("--keywordArray--");
    console.log(keywordsArray);
   Shiny.setInputValue("made_array", Array.from(keywordsArray))


      
      //nodeFiltered.style("fill", "red")
      // textFiltered.style("fill", "red")
      // .style("stroke", "red");
      //linkFiltered.style("fill", "red")
      //return keywordsArray;
    } 
    //magnify(keywordsArray);
    // else { 
    //   //converting keyword set to array
    //   keywordsArray = Array.from(keywordsArray);
    //   console.log("--keywordArray--");
    //   console.log(keywordsArray);
    //   //Shiny.onInputChange

      
    //   keywordsArray = [];
    //   keywordsArray = new Set();
      
    //   node.style("fill", function(d) { return color(d.isTriple); })
    //   text.style("fill", "black")
    //   .style("stroke", "black");
    //   return;
      
    // }
   // drawNetwork(bib, keywordsArray, author);
  }
 
  
  drawNetwork(bib, keywordsArray, author);
  
  var edgeList = innerChart.append("g").attr("transform", `translate(${[width/2 + margin/2, margin/2]})`);
      edgeList.append("text");

  var pub = innerChart.append("g").attr("transform", `translate(${[width/2 + margin/2, margin/2]})`);
      pub.append("text");

  var adjlist = [];

graph.links.forEach(function(d) {
    adjlist[d.source + "-" + d.target] = true;
    adjlist[d.target + "-" + d.source] = true;
    
            // console.log(keywordsArray);

});

function neigh(a, b) {
  return a == b || adjlist[a + "-" + b];
}
function adj(a, lk){
  lk.forEach(function(l){
    if (l.source.id === a || l.target.id === a) {
      return true;
    }else{
      return false;
    }
  })
}

  // var selected = false
  node.on("click",function(evt, d){
    var selectedID = d3.select(event.target).attr("id").split("_")[1];
    console.log("target: "+ d3.select(evt.target));
    var textID = "#text_" + selectedID;
    // node.style("fill-opacity", "0.2");

    // link
    //     .style('stroke', "#b8b8b8")
    //     .style("stroke-opacity", "0.3")
    //     .style("stroke-width", "0.5");
    
    // text
    //     .style("fill-opacity", "0.1")
    //     .style("stroke-opacity", "0.1");
    d3.select(textID)
      .style('stroke', "red")
      .style('fill', "red")
      .style("stroke-width", "0.7")
      .style("fill-opacity", 1)
      .style('stroke-opacity', 1);

    
      
    d3.select(evt.target)
          .style('fill-opacity', "0.9")
          .style('fill', "orange");

      
      //console.log(d.id);
      //console.log( node.filter(function(n) { return adj(n.id, currId); }));
      // node.filter(function(d) { adj(d.id, currId); }).style("fill", "yellow");
        //   node.style("opacity", function(o) {
        //       return neigh(index, o.id) ? 1 : 0.1;
        //   });
        //   node.style("fill", function(o) {
        //     return neigh(index, o.id) ? "yellow" : "black";
        // });
        //   text.style("fill", function(o) {
        //     return neigh(index, o.node) ? "yellow" : "black";
        //   });
        //   link.style("opacity", function(o) {
        //       return o.source == index || o.target == index ? 1 : 0.5;
        //   });
    lk = link.filter(l => l.source.id === d.id || l.target.id === d.id);
    // lk.each(function(d){
    //   console.log("in the loop");
    //   console.log(d.source);
    // });
    //node.filter(function(n) { return adj(n.id, lk); }).style("fill", "yellow");
    //console.log("lk"+lk[0]);
    // lk.each(function(l){
    //   node.filter(function(n) { return (n.id === l.source.id || n.id === l.target.id) }).style("fill", "yellow");
    //   text.filter(function(n) { return (n.id === l.source.id || n.id === l.target.id) }).style("fill", "yellow");
    //   // node.each(function(d){
    //   //   if (d.id === l.source.id || d.id === l.target.id){
    //   //     //console.log(d.id);
    //   //     d3.select(this).style('fill-opacity', 1)
    //   //     d3.select(this).style("fill", "red");
    //   //   }
    //   // })
    //   // text.each(function(d){
    //   //   if (d.id === l.source.id || d.id === l.target.id){
    //   //     //console.log(d.id);
    //   //     d3.select(this).style("stroke-opacity", 1);
    //   //     d3.select(this).style('fill-opacity', 1)
    //   //     d3.select(this).style("fill", "red");
    //   //   }
    //   // })
    // })
      
    //console.log("d: "+ evt.target.attr("selected"));
    console.log("selected: " + d3.select(evt.target).attr("selected"));
    if (d3.select(evt.target).attr("selected")){
      lk.style("stroke", "#999")
      .style("stroke-opacity", 0.6)
      .style("stroke-width", d => d.weight);
      d3.select(evt.target).attr("selected", !d3.select(evt.target).attr("selected"));
      keywordsArray.delete(d.keywords);
      d3.select(evt.target).style("fill", "steelblue");
        // text
        //   .style("fill", "black")
        //   .style("stroke", "black");
      } else {
        d3.select(evt.target).attr("selected", true);
        keywordsArray.add(d.keywords);
        d3.select(evt.target).style("fill", "red")
        lk
          .style('stroke', "#69b3b2")
          .style("stroke-opacity", "0.9")
          .style("stroke-width", 4)
        // textFiltered
        // .style("fill", "red")
        // .style("stroke", "red");
      }  
    
        
        
        console.log("--keywords Array--")
        
        console.log(keywordsArray);
    

        
      // link
      //   .filter(l => (l.source.id === d.id || l.target.id === d.id))
      //   .style('stroke', "#69b3b2")
      //   .style("stroke-opacity", "0.9")
      //   .style("stroke-width", 4)
    })
  // node.on("click", (evt, d) => {
    
    
  
  //   //console.log(d3.select(event.target).attr("selected"));
  //   // if(selected == true){
  //   //   console.log("selected is true? "+selected);
  //   //   d3.select(evt.target).style("fill", "red");
  //   // }
  //   // else{
  //   //   d3.select(evt.target).style("fill", function(d) { return color(d.isTriple); });
  //   // }
    
  //   // bib.forEach(function(b){if (b.keywords_id.includes(d.id)){
  //   //   pub.select("text")
  //   //   .attr("x", 15)
  //   //   .attr("y", 5)
  //   //   .text("Title: ")
  //   //   .style("stroke", "green")
  //   //   .append('tspan')
  //   //   .text(b.bib_title)
  //   //   .style("stroke-width", "0.7")
  //   //   .style("stroke", "blue")
  //   //   .append('tspan')
      
  //   //   .attr("x", 15)
  //   //   .attr("y", 30)
  //   //   .text("eprint_url: ")
  //   //   .style("stroke", "green")
  //   //   .append('tspan')
  //   //   .text(b.eprint_url)
  //   //   .style("stroke-width", "0.7")
  //   //   .style("stroke", "blue");
      
      
  //   // }
  // })

    
    
    
    
   



  node.on("mouseenter", (evt, d) => {
    
    // bib.forEach(function(b){if (b.keywords_id.includes(d.id)){
    //   pub.select("text")
    //   .attr("x", 15)
    //   .attr("y", 5)
    //   .text("Title: ")
    //   .style("stroke", "green")
    //   .append('tspan')
    //   .text(b.bib_title)
    //   .style("stroke-width", "0.7")
    //   .style("stroke", "blue")
    //   .append('tspan')
      
      
    //   .attr("x", 15)
    //   .attr("y", 30)
    //   .text("eprint_url: ")
    //   .style("stroke", "green")
    //   .append('tspan')
    //   .text(b.eprint_url)
    //   .style("stroke-width", "0.7")
    //   .style("stroke", "blue");
      
      
    // }
  // });

    //lk = link.filter(l => l.source.id === d.id || l.target.id === d.id);
    
    // lk.each(function(d,i){
    //   // console.log(d)
    //   edgeList.select("text")
    //   .attr("x", 15)
    //   .attr("y", 50)
    //   .text("Source: ")
    //   .style("stroke", "red")
    //   .append('tspan')
    //   .text(d.source_keyword)
    //   .style("stroke-width", "0.7")
    //   .style("stroke", "yellow")
    //   .append('tspan')
      
    //   .attr("x", 15)
    //   .attr("y", 70)
    //   .text("Target: ")
    //   .style("stroke", "red")
    //   .append('tspan')
    //   .text(d.target_keyword)
    //   .style("stroke-width", "0.7")
    //   .style("stroke", "steelblue");
    //   //textArr.push("Source: " + d.source_keyword + "," + " Target: " + d.target_keyword);
    // })
    // var selectedID = d3.select(event.target).attr("id").split("_")[1];
    // var textID = "#text_" + selectedID;
    
    // text
    //     .style("fill-opacity", "0.1")
    //     .style("stroke-opacity", "0.1");
    // d3.select(textID)
    //   .style('stroke', "red")
    //   .style("stroke-width", "0.7")
    //   .style('stroke-opacity', "1");

    // node.style("fill-opacity", "0.2");

      
    // d3.select(evt.target)
    //       .style('fill-opacity', "0.9")
    //       .style('fill', "orange");

    // nodeFiltered = node.filter(n => (n.group.some(element => {
    //   return d.group.includes(element);})));
    // textFiltered = text.filter(t => (t.group.some(element => {
    //   return d.group.includes(element);})));
  
    // textFiltered
    // .style('stroke', 'orange')
    // .style("stroke-opacity", "1");
    // nodeFiltered
    // .style('fill', 'orange')
    // .style("fill-opacity", "0.6");

  //   node.style('fill', function (n) { return n.group.some(element => {
  // return d.group.includes(element);}) ? 'orange' : 'blue';})
    
  //     link
  //       .style('stroke', function (link_d) { return link_d.group.some(element => {
  // return d.group.includes(element);}) ? '#69b3b2' : '#b8b8b8';})
  //       .style('stroke-opacity', function (link_d) { return link_d.group.some(element => {
  // return d.group.includes(element);}) ? '0.9' : '0.2';})
  //       .style('stroke-width', function (link_d) { return d.isTriple && link_d.group.some(element => {
  // return d.group.includes(element);}) ? 4 : 1;});
        
      // link
      //   .filter(l => (l.source.id === d.id || l.target.id === d.id))
      //   .style('stroke', "red")
      //   .style("stroke-opacity", "0.9")
      //   .style("stroke-width", d => d.weight)
        
        })
    // .on("mouseleave", (evt, d) => {
    //   if(d3.select(this).classed("selected")){
    //   var selectedID = d3.select(event.target).attr("id").split("_")[1];
    // var textID = "#text_" + selectedID;
    //   d3.select(textID)
    //     .style("stroke-width", "0.7")
    //     .style("stroke", "black")
    //     .style('stroke-opacity', "0.7");
    //   d3.select(evt.target)
    //     .style('fill-opacity', "0.7")
    //     .style("fill", function(d) { return color(d.isTriple); });
    //   node
    //     .style("fill-opacity", "0.7")
    //     .style("fill", function(d) { return color(d.isTriple);});
    //   text
    //     .style("stroke", "black")
    //     .style("fill-opacity", "0.7")
    //     .style("stroke-opacity", "0.7");

    //   link
    //       .style('stroke', "#999")
    //       .style("stroke-width", d => d.weight)
    //   } else{

    //   }
      
    // });

  function drag(simulation) {    
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
  
      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }
  
      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      } 
      return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("start end", dragended);
  }
  // labelSim.on("tick", () => {
  //   labelSim.alphaTarget(0.3).restart();
  //   text.attr("x", d => d.x)
  //       .attr("y", d => d.y);
  // })
  simulation.on("tick", function() {
    //labelSim.alphaTarget(0.3).restart();
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });
  node.attr("cx", d => d.x)
      .attr("cy", d => d.y);
    //text.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  text
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });
  d3.select("#clearButton").on("click", function() {
    //keywordsArray = [];
    keywordsArray = new Set();
    updateNodes(keywordsArray);
    // node.style("fill", function(d) { return color(d.isTriple); })
    d3.select("#chart").selectAll(".texts").style("stroke", "black").style("fill", "black")
    // .style("stroke", "black");
  d3.select('.brush')
  .call(brush.move, null);
});
function enableBrush(){
  d3.select("#chart").select(".brush")
    .call(brush);
}
d3.select("#brushButton").on("click", function() {
  enableBrush();
})
d3.select("#textToggle").on("click", function() {
  showText(text);
})
d3.select("#removeButton").on("click", function() {
  d3.select('.brush').remove();
})
}
draw();
//end of function

  d3.select("#clusterButton").on("click", function() {
          netClustering.cluster(graph.nodes, graph.links);

          link
            .transition()
            .duration(500)
            .style("stroke", function(d) {
              return color(d.isTriple);
            });
        });

  function onCheckGroupInABox() {
    simulation.stop();
    useGroupInABox = d3.select("#checkGroupInABox").property("checked");
    simulation.force("isTriple").enableGrouping(useGroupInABox)

    simulation.alphaTarget(0.5).restart();
  }

});
