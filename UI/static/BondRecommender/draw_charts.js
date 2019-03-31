new Chart(document.getElementById("myChart"), {
  type: "doughnut",
  data: {
    labels: ["red", "blue", "yellow"],
    datasets: [
      {
        data: [300, 50, 100],
        backgroundColor: [
          "rgb(255, 99, 132)",
          "rgb(54, 162, 235)",
          "rgb(255, 205, 86)"
        ]
      }
    ]
  }
});