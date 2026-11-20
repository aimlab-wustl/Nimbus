# Vivado batch build of the one-syndrome-per-shot decoder board design (pt_board + pt_top), from the repository
# (hardware/README.md, "Build a bitstream").
#
#   vivado -mode batch -nojournal -log <out>/vivado.log -source hardware/build/build_pt_board.tcl -tclargs key=value ...
#
# Keys (defaults in brackets):
#   d [7]  NS [16]  K [2]  B [16]        decoder size (hardware/gen/gen_pt.py)
#   p [0.12]  bmin [0.1]                 power-on tables only; the host writes the real ones at run time
#   target [zu7ev]                       rfsoc4x2 | zu7ev | zcu104 (hardware/build/board_cfg.tcl)
#   part [from target]  board_part [from target; "none" = no board files]
#   board_repo [""]                     folder with board files (set_param board.repoPaths; nothing is written to the Vivado
#                                        install). Without board files the PS gets a minimal configuration (enough for PYNQ).
#   dec_mhz [300]  axi_mhz [100]         decoder clock (MMCM from PL clock 0) and AXI / PL clock 0
#   pl0_board [100 for rfsoc4x2*, else ""]  PL clock 0 as the board really makes it. PYNQ applies the .hwh dividers to the
#                                        board's own PLL, so Vivado's PL clock 0 (96.97 MHz here) is not what the board
#                                        runs; the driver sets PL clock 0 to this value, the MMCM is set so that dec_mhz is
#                                        the decoder clock on the board, and timing is analysed at that clock.
#   out [hardware/build/out/<name>]  name [pt_d<d>_ns<NS>]
#   gen [generate]                       folder with a pre-generated pt_core.sv / pt_params.svh / pt_info.json (a bare
#                                        name means hardware/cores/<name>), or generate
#   python [$PT_PYTHON or python]        interpreter for the generator (needs numpy, numba and the nimbus package)
#   jobs [4]  strategy [default]         launch_runs jobs; implementation strategy (e.g. Performance_Explore)
#   proj [<out>/proj]                    Vivado project folder
#   stop [bitstream]                     bd (block design only) | synth | bitstream
# Outputs in <out>: <name>.bit, <name>.hwh (PYNQ needs both, same base name), <name>.xsa, <name>.json (parameters, git
# commit, Vivado version, clocks, WNS, utilization), reports/, gen/ (RTL), proj/ (Vivado project), BUILD_OK or BUILD_FAILED.
# IP versions are not pinned: each IP is taken at the newest version this Vivado has (the published bitstreams were built with
# Vivado 2022.1; the script also runs on 2024.2).

set here [file normalize [file dirname [info script]]]
set root [file normalize [file join $here .. ..]]
source [file join $here board_cfg.tcl]

# ------------------------------------------------------------------------------------------------ arguments
array set A {d 7 NS 16 K 2 B 16 p 0.12 bmin 0.1 target zu7ev part "" board_part "" dec_mhz 300 axi_mhz 100 out "" name ""
             gen generate python "" jobs 4 strategy default stop bitstream syn_aw 14 res_aw 15 board_repo "" pl0_board ""}
# key=value pairs or space-separated "key value" pairs (Windows cmd splits batch arguments at '=', ',' and ';', so
# "-tclargs d 7 ns 16" is accepted too). Keys are case-insensitive.
array set A {proj ""}
foreach k [array names A] { set KEY([string tolower $k]) $k }
set av $argv
while {[llength $av]} {
    set kv [lindex $av 0]; set av [lrange $av 1 end]
    set i [string first = $kv]
    if {$i > 0} {
        set k [string range $kv 0 [expr {$i - 1}]]; set v [string range $kv [expr {$i + 1}] end]
    } else {
        if {![llength $av]} { error "key '$kv' without a value" }
        set k $kv; set v [lindex $av 0]; set av [lrange $av 1 end]
    }
    if {![info exists KEY([string tolower $k])]} { error "unknown key '$k'" }
    set A($KEY([string tolower $k])) $v
}
set tgt [pt_target $A(target)]
if {$A(part) eq ""} { set A(part) [dict get $tgt part] }
if {$A(board_part) eq ""} { set A(board_part) [dict get $tgt board_part] }
if {$A(board_part) eq "none"} { set A(board_part) "" }
if {$A(name) eq ""} { set A(name) "pt_d$A(d)_ns$A(NS)" }
if {$A(pl0_board) eq "" && [string match rfsoc4x2* $A(target)]} { set A(pl0_board) 100 }
if {$A(out) eq ""} { set A(out) [file join $root hardware build out $A(name)] }
set out [file normalize $A(out)]
file mkdir $out [file join $out reports]
file delete -force [file join $out BUILD_OK] [file join $out BUILD_FAILED]
set t0 [clock seconds]
proc fail {msg} {
    global out
    set f [open [file join $out BUILD_FAILED] w]; puts $f $msg; close $f
    set f [open [file join $out BUILD_FAILED.txt] w]; puts $f $msg; close $f
    puts "BUILD_FAILED: $msg"
    exit 1
}

# newest installed version of an IP
proc ipv {name} {
    set defs [lsort -dictionary [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
    if {[llength $defs] == 0} { fail "IP $name not found in this Vivado" }
    return [lindex $defs end]
}

# git commit of the sources
set git_hash unknown; set git_dirty unknown
catch { set git_hash [string trim [exec git -C $root rev-parse HEAD]] }
catch { set git_dirty [expr {[string length [string trim [exec git -C $root status --porcelain -- hardware src]]] > 0}] }
puts "BUILD $A(name): d=$A(d) NS=$A(NS) K=$A(K) B=$A(B) part=$A(part) board=$A(board_part) dec=$A(dec_mhz) MHz axi=$A(axi_mhz) MHz git=$git_hash dirty=$git_dirty"

# ------------------------------------------------------------------------------------------------ RTL generation
if {$A(gen) eq "generate"} {
    set gdir [file join $out gen]
    file mkdir $gdir
    set py $A(python)
    if {$py eq "" && [info exists ::env(PT_PYTHON)]} { set py $::env(PT_PYTHON) }
    if {$py eq ""} { set py python }
    # Vivado points PYTHONHOME / PYTHONPATH at its own Python 3.8; the generator needs the project's interpreter
    # (cleared for the call only: Vivado itself needs them, e.g. to load IP integrator)
    set saved {}
    foreach v {PYTHONHOME PYTHONPATH} { if {[info exists ::env($v)]} { lappend saved $v $::env($v); unset ::env($v) } }
    set rc [catch {exec $py [file join $root hardware gen gen_pt.py] --d $A(d) --NS $A(NS) --K $A(K) --B $A(B) --p $A(p) --bmin $A(bmin) \
                   --out $gdir} msg]
    foreach {v val} $saved { set ::env($v) $val }
    if {$rc} { fail "RTL generation failed: $msg" }
} else {
    set gdir $A(gen)
    # a bare name means hardware/cores/<name>
    if {[file pathtype $gdir] eq "relative" && [llength [file split $gdir]] == 1} { set gdir [file join $root hardware cores $gdir] }
    if {[file pathtype $gdir] eq "relative"} { set gdir [file join $root $gdir] }
    set gdir [file normalize $gdir]
}
foreach f {pt_core.sv pt_params.svh pt_info.json} {
    if {![file exists [file join $gdir $f]]} { fail "missing [file join $gdir $f]" }
}

# ------------------------------------------------------------------------------------------------ project and block design
# project folder: <out>/proj unless proj= is given
if {$A(proj) ne ""} { set proj [file normalize $A(proj)] } else { set proj [file join $out proj] }
file delete -force $proj
if {$A(board_repo) ne ""} { set_param board.repoPaths [list [file normalize $A(board_repo)]] }
create_project pt_proj $proj -part $A(part) -force
set use_preset 0
if {$A(board_part) ne ""} {
    if {[llength [get_board_parts -quiet $A(board_part)]]} {
        set_property board_part $A(board_part) [current_project]
        set use_preset 1
    } else {
        puts "WARNING: board part $A(board_part) not installed; building with the part only and a minimal PS"
    }
}
set rtl [file join $root hardware rtl]
add_files -norecurse [list [file join $rtl tp_popcnt.sv] [file join $rtl tp_lut_rom.sv] [file join $rtl pt_xoshiro.sv] \
    [file join $rtl pt_thr.sv] [file join $rtl pt_units.sv] [file join $rtl pt_top.sv] [file join $rtl pt_board.sv] [file join $rtl pt_board_top.v] \
    [file join $gdir pt_core.sv] [file join $gdir pt_params.svh]]
set_property include_dirs [list $gdir] [current_fileset]
update_compile_order -fileset sources_1

create_bd_design pt_bd
set ps [create_bd_cell -type ip -vlnv [ipv zynq_ultra_ps_e] ps]
pt_ps_config $ps $A(axi_mhz) $use_preset
# MMCM request in Vivado's view of PL clock 0, so that the board (PL clock 0 = pl0_board) runs the decoder at dec_mhz
set pl0_viv [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
set dec_req [expr {$A(pl0_board) eq "" ? $A(dec_mhz) : $A(dec_mhz) * $pl0_viv / $A(pl0_board)}]
set ck [create_bd_cell -type ip -vlnv [ipv clk_wiz] clk_dec]
set ra [create_bd_cell -type ip -vlnv [ipv proc_sys_reset] rst_axi]
set rd [create_bd_cell -type ip -vlnv [ipv proc_sys_reset] rst_dec]
set sc [create_bd_cell -type ip -vlnv [ipv smartconnect] axi_sc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc
if {[catch {create_bd_cell -type module -reference pt_board_top pt_0} pt]} { fail "module reference pt_board_top: $pt" }
set_property -dict [list CONFIG.SYN_AW $A(syn_aw) CONFIG.RES_AW $A(res_aw)] $pt
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins clk_dec/clk_in1] [get_bd_pins rst_axi/slowest_sync_clk] \
    [get_bd_pins axi_sc/aclk] [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins pt_0/s_axi_aclk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_axi/ext_reset_in] [get_bd_pins rst_dec/ext_reset_in]
connect_bd_net [get_bd_pins clk_dec/clk_out1] [get_bd_pins rst_dec/slowest_sync_clk] [get_bd_pins pt_0/dclk]
connect_bd_net [get_bd_pins clk_dec/locked] [get_bd_pins rst_dec/dcm_locked]
connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn] [get_bd_pins pt_0/s_axi_aresetn] [get_bd_pins axi_sc/aresetn]
connect_bd_net [get_bd_pins rst_dec/peripheral_aresetn] [get_bd_pins pt_0/dresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_sc/M00_AXI] [get_bd_intf_pins pt_0/s_axi]
# MMCM: input from PL clock 0 (a global buffer), one output at the decoder clock, no reset port
set_property -dict [list CONFIG.PRIM_SOURCE {Global_buffer} CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ [format %.3f $dec_req]] $ck
assign_bd_address
if {[catch {validate_bd_design} msg]} { fail "validate_bd_design: $msg" }
save_bd_design
set bd [get_files pt_bd.bd]
set wrapper [make_wrapper -files $bd -top]
add_files -norecurse $wrapper
set_property top pt_bd_wrapper [current_fileset]
generate_target all $bd
# achieved clocks (MMCM output may differ slightly from the request)
# achieved decoder clock: the FREQ_HZ the MMCM output propagates (CLKOUT1_ACTUAL_FREQ is empty in some versions)
set dec_actual [expr {[get_property CONFIG.FREQ_HZ [get_bd_pins clk_dec/clk_out1]] / 1.0e6}]
set axi_actual [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
# on the board both scale with PL clock 0
if {$A(pl0_board) ne ""} {
    set dec_actual [expr {$dec_actual * $A(pl0_board) / $axi_actual}]; set axi_actual [expr {double($A(pl0_board))}]
}
set addr_seg [lindex [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] 0]
set base [get_property OFFSET $addr_seg]; set range [get_property RANGE $addr_seg]
puts "CLOCKS dec_actual=$dec_actual MHz axi_actual=$axi_actual MHz; pt_0 at $base range $range"
if {$A(stop) eq "bd"} { set f [open [file join $out BUILD_OK] w]; puts $f bd; close $f; exit 0 }

# clock-domain crossings of pt_board (request/acknowledge tunnel): data held for >= 2 decoder cycles before use
set per_dec [expr {1000.0 / $dec_actual}]
set xdc [file join $out pt_board_cdc.xdc]
set f [open $xdc w]
puts $f "# generated by hardware/build/build_pt_board.tcl: crossings of pt_board (see hardware/rtl/pt_board.sv)"
puts $f "set_max_delay -datapath_only -from \[get_cells -hier -filter {NAME =~ */u_brd/cdcA_* && IS_SEQUENTIAL}\] [format %.3f [expr {2 * $per_dec}]]"
puts $f "set_max_delay -datapath_only -from \[get_cells -hier -filter {NAME =~ */u_brd/cdcD_* && IS_SEQUENTIAL}\] [format %.3f [expr {2 * $per_dec}]]"
puts $f "# runtime threshold table: write data held two decoder cycles before the idle-only write strobe (hardware/rtl/pt_top.sv)"
puts $f "set_multicycle_path -setup 2 -from \[get_cells -hier -filter {NAME =~ */u_pt/twd_r_reg*}\]"
puts $f "set_multicycle_path -hold 1 -from \[get_cells -hier -filter {NAME =~ */u_pt/twd_r_reg*}\]"
if {$A(pl0_board) ne ""} {
    puts $f "# PL clock 0 at the board's value (the PS IP defines it at Vivado's value); the MMCM clocks are derived from it"
    puts $f "create_clock -period [format %.4f [expr {1000.0 / $A(pl0_board)}]] -name clk_pl_0 \[get_pins -hierarchical *PS8_i/PLCLK\\\[0\\\]\]"
}
puts $f "# compressed bitstream (the board's file system is nearly full)"
puts $f "set_property BITSTREAM.GENERAL.COMPRESS TRUE \[current_design\]"
close $f
add_files -fileset constrs_1 -norecurse $xdc
set_property used_in_synthesis false [get_files $xdc]
set_property PROCESSING_ORDER LATE [get_files $xdc]

# ------------------------------------------------------------------------------------------------ synthesis, implementation
launch_runs synth_1 -jobs $A(jobs)
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { fail "synthesis failed (see proj/pt_proj.runs/synth_1/runme.log)" }
if {$A(stop) eq "synth"} { set f [open [file join $out BUILD_OK] w]; puts $f synth; close $f; exit 0 }
if {$A(strategy) ne "default"} { set_property strategy $A(strategy) [get_runs impl_1] }
launch_runs impl_1 -to_step write_bitstream -jobs $A(jobs)
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { fail "implementation failed (see proj/pt_proj.runs/impl_1/runme.log)" }

# ------------------------------------------------------------------------------------------------ results
open_run impl_1
set rep [file join $out reports]
report_utilization -file [file join $rep util_impl.rpt]
report_utilization -hierarchical -hierarchical_depth 4 -file [file join $rep util_hier_impl.rpt]
report_timing_summary -max_paths 10 -file [file join $rep timing_impl.rpt]
report_clock_interaction -file [file join $rep clock_interaction.rpt]
report_power -file [file join $rep power_impl.rpt]
catch { report_cdc -file [file join $rep cdc.rpt] }
# per-clock setup slack of the decoder clock and of the AXI clock
proc clk_wns {pin} {
    set c [get_clocks -quiet -of_objects [get_pins -quiet $pin]]
    if {$c eq ""} { return [list "" 0 ""] }
    set p [get_timing_paths -quiet -setup -max_paths 1 -group $c]
    set s [expr {$p eq "" ? "" : [get_property SLACK $p]}]
    return [list [get_property NAME $c] [get_property PERIOD $c] $s]
}
lassign [clk_wns pt_bd_i/pt_0/dclk] dec_clk dec_per dec_wns
lassign [clk_wns pt_bd_i/pt_0/s_axi_aclk] axi_clk axi_per axi_wns
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
set fmax_dec [expr {$dec_wns eq "" ? "" : [format %.1f [expr {1000.0 / ($dec_per - $dec_wns)}]]}]
set ut [report_utilization -return_string]
proc ucell {txt key} {
    if {[regexp -line "^\\|\\s+${key}\\*?\\s+\\|\\s+(\\d+(?:\\.\\d+)?)\\s+\\|" $txt -> v]} { return $v }
    return ""
}
set lut [ucell $ut "CLB LUTs"]; set ff [ucell $ut "CLB Registers"]; set bram [ucell $ut "Block RAM Tile"]
set dsp [ucell $ut "DSPs"]; set lutram [ucell $ut "LUT as Memory"]
puts "RESULT $A(name): WNS $wns ns (decoder clock $dec_clk: period $dec_per ns, WNS $dec_wns ns, Fmax $fmax_dec MHz; AXI clock WNS $axi_wns ns), WHS $whs ns; LUT $lut (LUTRAM $lutram), FF $ff, BRAM $bram, DSP $dsp"

set runs [get_property DIRECTORY [get_runs impl_1]]
set bit [lindex [glob -nocomplain [file join $runs *.bit]] 0]
if {$bit eq ""} { fail "no bitstream" }
file copy -force $bit [file join $out $A(name).bit]
set hwh [lindex [glob -nocomplain [file join $proj pt_proj.gen sources_1 bd pt_bd hw_handoff pt_bd.hwh] \
                                  [file join $proj pt_proj.srcs sources_1 bd pt_bd hw_handoff pt_bd.hwh]] 0]
if {$hwh eq ""} { fail "no .hwh found" }
file copy -force $hwh [file join $out $A(name).hwh]
catch { write_hw_platform -fixed -include_bit -force [file join $out $A(name).xsa] }

set info [open [file join $gdir pt_info.json] r]; set pinfo [string trim [read $info]]; close $info
set f [open [file join $out $A(name).json] w]
puts $f "{"
foreach k [lsort [array names A]] { puts $f "  \"arg_$k\": \"[string map {\\ \\\\ \" \\\"} $A($k)]\"," }
puts $f "  \"git\": \"$git_hash\", \"git_dirty\": \"$git_dirty\", \"vivado\": \"[version -short]\","
puts $f "  \"pl0_board_mhz\": \"$A(pl0_board)\", \"pl0_vivado_mhz\": $pl0_viv,"
puts $f "  \"dec_mhz_actual\": $dec_actual, \"axi_mhz_actual\": $axi_actual, \"base\": \"$base\", \"range\": \"$range\","
puts $f "  \"wns\": $wns, \"whs\": $whs, \"dec_clock\": \"$dec_clk\", \"dec_period\": $dec_per, \"dec_wns\": \"$dec_wns\","
puts $f "  \"dec_fmax_mhz\": \"$fmax_dec\", \"axi_wns\": \"$axi_wns\","
puts $f "  \"lut\": \"$lut\", \"lutram\": \"$lutram\", \"ff\": \"$ff\", \"bram\": \"$bram\", \"dsp\": \"$dsp\","
puts $f "  \"build_s\": [expr {[clock seconds] - $t0}],"
puts $f "  \"pt_info\": $pinfo"
puts $f "}"
close $f
# deploy/: only what the board needs (bit + hwh with the same base name, build info, driver)
set dep [file join $out deploy]
file mkdir $dep
foreach x [list $A(name).bit $A(name).hwh $A(name).json] { file copy -force [file join $out $x] $dep }
file copy -force [file join $root hardware host pt_board.py] $dep
set f [open [file join $out BUILD_OK.txt] w]; puts $f "$A(name) wns=$wns"; close $f
set f [open [file join $out BUILD_OK] w]; puts $f "$A(name) wns=$wns"; close $f
puts "BUILD_DONE $A(name) in [expr {[clock seconds] - $t0}] s -> $out"
