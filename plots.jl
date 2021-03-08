using Plots

function gen_plots()
    # gr(size=(500,400))
    println("*************** generating plots ******************")

    plt = plot(F_out, label="F_hat", color="blue", linewidth = 2)
    plot!(F_gt_data, label="F_gt", linestyle=:dash, color = "blue", linewidth = 2)

    plot!(α_out, label="α_hat", color="green", linewidth = 2)
    plot!(α_gt_data, label="α_gt", color="green", linewidth = 2, linestyle=:dash)
    title!("Forward sensitivity (analytic)")
    xlabel!("epochs")
    ylabel!("parameters")

    display(plt)
    savefig(plt, "./figures/analytic_two_params_pendulum_iter$(n_epchs)_lr$(lr).png")
end
