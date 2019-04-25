var drawGraph1 = function(isin_code) {
    var labels1 = "{{ bond_dates|safe }}";
    var labels1 = labels1.split("|");
    var data1 = "{{ bond_OAS|safe }}";
    var data1 = data1.split("|").map(parseFloat);
    var label1 = isin_code;
    new Chart(document.getElementById("myChart"), {
        type: "line",
        data: {
            labels: labels1,
            datasets: [
                {
                    label: label1,
                    data: data1,
                    borderColor: "rgb(185, 217, 235)",
                    fill: false
                }
            ]
        },
        options: {
            title: {
                display: true,
                text: 'OAS spread: ' + String(label1),
                fontSize: 16
            }
        }
    });
};
var drawGraph = function(isin_code) {
    var labels1 = "{{ bond_dates|safe }}";
    var labels1 = labels1.split("|");
    var data1 = "{{ bond_OAS|safe }}";
    var data1 = data1.split("|").map(parseFloat);
    var data2 = "{{ bond_OAS2|safe }}";
    var data2 = data2.split("|").map(parseFloat);
    var label1 = "{{ sec_id }}";
    var label2 = isin_code;
    new Chart(document.getElementById("myChart"), {
        type: "line",
        data: {
            labels: labels1,
            datasets: [
                {
                    label: label1,
                    data: data1,
                    borderColor: "rgb(185, 217, 235)",
                    fill: false
                },
                {
                    label: label2,
                    data: data2,
                    borderColor: "rgb(162, 10, 53)",
                    fill: false
                }
            ],
        },
        options: {
            title: {
                display: true,
                text: 'OAS spread: ' + String(label1) + ' vs ' + String(label2),
                fontSize: 16
            }
        }
    });
};

// new Chart(document.getElementById("myChart"), {
//   type: "doughnut",
//   data: {
//     labels: ["red", "blue", "yellow"],
//     datasets: [
//       {
//         data: [300, 50, 100],
//         backgroundColor: [
//           "rgb(255, 99, 132)",
//           "rgb(54, 162, 235)",
//           "rgb(255, 205, 86)"
//         ]
//       }
//     ]
//   }
// });
//
// var drawGraph = function(data){
//   var ctx = document.getElementById('graph').getContext('2d');
//   var myChart = new Chart(ctx, {
//     type: 'line',
//     data: { labels: data[0],
//     datasets: [{ label:'temperature', data:data[1] },
//                { label:'humidity', data:data[2] }]
//     }
//   });
// };
//
// window.onload=function () {
//     var data = [['12:00', '13:00', '14:00', '15:00', '16:00'],
//                 [22, 23, 21, 20, 19],
//                 [55, 50, 45, 43, 42]]
//     drawGraph(data);
// };