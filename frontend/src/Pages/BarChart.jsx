import React, { useEffect, useRef } from "react";
import { Chart } from "chart.js/auto";

const BarChart = ({ data }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    let chartInstance = chartRef.current?.chartInstance;

    if (chartInstance) {
      // Destroy existing chart instance if it exists
      chartInstance.destroy();
    }

    const ctx = chartRef.current.getContext("2d");

    const newChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: Object.keys(data),
        datasets: [
          {
            label: "Class Probabilities",
            data: Object.values(data).map((value) => value * 100), // Multiply each value by 100
            backgroundColor: [
              "rgba(255, 99, 132, 0.2)",
              "rgba(54, 162, 235, 0.2)",
              "rgba(255, 206, 86, 0.2)",
              "rgba(75, 192, 192, 0.2)",
            ],
            borderColor: [
              "rgba(255, 99, 132, 1)",
              "rgba(54, 162, 235, 1)",
              "rgba(255, 206, 86, 1)",
              "rgba(75, 192, 192, 1)",
            ],
            borderWidth: 1,
          },
        ],
      },
      options: {
        scales: {
          x: {
            beginAtZero: true,
          },
          y: {
            beginAtZero: true,
            ticks: {
              callback: (value) => `${value}%`, // Add '%' symbol to each y-axis tick
            },
          },
        },
      },
    });

    // Save the chart instance to the ref
    chartRef.current.chartInstance = newChart;

    // Cleanup: Destroy the chart instance when the component unmounts
    return () => {
      chartInstance = chartRef.current?.chartInstance;
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, [data]);

  return (
    <canvas
      className="w-full h-full max-w-5xl max-h-400"
    //   style={{ backgroundColor: "#f5f5f5" }}
      ref={chartRef}
    />
  );
};

export default BarChart;
