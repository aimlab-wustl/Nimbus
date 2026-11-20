# Target settings for build_pt_board.tcl and build_ptw_board.tcl. Everything that differs between the RFSoC 4x2 and the test targets lives
# in this file (part, board files, PS configuration). Nothing else in the flow names a board.
#
#   rfsoc4x2     Real Digital RFSoC 4x2, XCZU48DR-2, minimal PS (no board files needed; used for all published builds)
#   rfsoc4x2_bf  the same with the Real Digital board files (PS preset); pass board_repo=<folder> if they are not
#                installed in Vivado (set_param board.repoPaths)
#   zu7ev     generic Zynq UltraScale+ XCZU7EV-2 with a minimal PS (covered by the free Vivado edition; useful to test the flow)
#   zcu104    ZCU104 evaluation board (XCZU7EV-2) with its board preset, if those board files are installed

proc pt_target {target} {
    switch -- $target {
        rfsoc4x2 { return [dict create part xczu48dr-ffvg1517-2-e board_part ""] }
        rfsoc4x2_bf { return [dict create part xczu48dr-ffvg1517-2-e board_part realdigital.org:rfsoc4x2:part0:1.0] }
        zu7ev    { return [dict create part xczu7ev-ffvc1156-2-e board_part ""] }
        zcu104   { return [dict create part xczu7ev-ffvc1156-2-e board_part xilinx.com:zcu104:part0:1.0] }
        default  { error "unknown target '$target' (rfsoc4x2, rfsoc4x2_bf, zu7ev, zcu104)" }
    }
}

# Zynq UltraScale+ PS: board preset when the board part is set on the project (and its files are installed), then the settings
# this design needs: only M_AXI_HPM0_FPD (128-bit default width, as PYNQ expects), PL clock 0 at axi_mhz, PL reset 0, no
# interrupts. HPM1 / HPM0_LPD are switched off so that no AXI clock is left unconnected.
proc pt_ps_config {ps axi_mhz use_preset} {
    if {$use_preset} {
        if {[catch {apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps} err]} {
            puts "WARNING: board preset not applied ($err); using the minimal PS configuration"
        }
    }
    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $axi_mhz \
    ] $ps
}
