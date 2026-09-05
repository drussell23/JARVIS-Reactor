<#
.SYNOPSIS
    Read or set the NVIDIA "CUDA - Sysmem Fallback Policy" without the
    Control Panel, through NVAPI's driver-settings (DRS) interface.

.DESCRIPTION
    The WDDM driver backs CUDA allocations with HOST memory once the card
    is full. On this WSL2 box torch measured 61.75 GiB "allocated" on a
    32 GiB card; every byte past the card is Windows commit on vmmemWSL,
    and that spill is what took the desktop down on 2026-09-04
    (see memory_guard.DEFAULT_CUDA_ALLOCATOR_FRACTION and the sentinel).

    The policy is a DRS setting on the driver's BASE (global) profile:

        id 0x10ECECC9   0 = driver default (fallback ON for CUDA apps)
                        1 = Prefer No Sysmem Fallback
                        2 = Prefer Sysmem Fallback

    It is stored in %ProgramData%\NVIDIA Corporation\Drs\nvdrsdb0.bin,
    which is user-writable, so no elevation is needed -- the Control Panel
    itself runs unelevated and writes the same file through the same API.
    New CUDA contexts pick the value up immediately; running ones do not.

.EXAMPLE
    .\nv_sysmem_fallback.ps1 -Get
    .\nv_sysmem_fallback.ps1 -Set 1
    # then verify from WSL: bash scripts/host/probe_sysmem_fallback.sh

.NOTES
    NVAPI exports one symbol, nvapi_QueryInterface(id); everything else is
    a function pointer looked up by a published 32-bit id. The NVDRS_SETTING
    struct is handled as a raw 12,320-byte buffer (version 0x00013020 =
    sizeof | 1<<16) so no marshalling can drift: field offsets are
    settingId 4100, settingType 4104, settingLocation 4108,
    isCurrentPredefined 4112, isPredefinedValid 4116, u32CurrentValue 8220.
    A wrong id makes QueryInterface return null and the script abort
    before touching the database; NVAPI validates the struct on SetSetting.
#>
[CmdletBinding(DefaultParameterSetName = "Get")]
param(
    [Parameter(ParameterSetName = "Get")] [switch]$Get,
    [Parameter(ParameterSetName = "Set", Mandatory = $true)] [ValidateSet(0, 1, 2)] [int]$Set
)

$ErrorActionPreference = "Stop"

$cs = @"
using System;
using System.Runtime.InteropServices;

public static class NvDrs
{
    [DllImport("nvapi64.dll", EntryPoint = "nvapi_QueryInterface", CallingConvention = CallingConvention.Cdecl)]
    static extern IntPtr QueryInterface(uint id);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnVoid();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnOutPtr(out IntPtr h);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnPtr(IntPtr h);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnPtrOutPtr(IntPtr h, out IntPtr p);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnGetSetting(IntPtr h, IntPtr p, uint id, IntPtr setting);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate int FnSetSetting(IntPtr h, IntPtr p, IntPtr setting);

    static T Fn<T>(uint id, string name) where T : class
    {
        IntPtr p = QueryInterface(id);
        if (p == IntPtr.Zero) throw new Exception("NVAPI: " + name + " (0x" + id.ToString("X8") + ") not found");
        return Marshal.GetDelegateForFunctionPointer(p, typeof(T)) as T;
    }

    public const uint SETTING_ID = 0x10ECECC9;   // CUDA - Sysmem Fallback Policy
    const int SETTING_SIZE = 12320;
    const uint SETTING_VER = 0x00013020;          // NVDRS_SETTING_VER1
    const int OFF_ID = 4100, OFF_TYPE = 4104, OFF_LOC = 4108, OFF_ISCURPRE = 4112, OFF_ISPREVALID = 4116, OFF_CUR = 8220;
    const int NVAPI_SETTING_NOT_FOUND = -160;   // nvapi_lite_common.h; -163 is PROFILE_NOT_FOUND

    static void Check(int rc, string what)
    {
        if (rc == 0) return;
        string why = "";
        // Measured 2026-09-04 on driver 32.0.16.1062 as a standard user:
        // SetSetting on the hidden "CUDA - Force P2 State" id returns -137,
        // and on 0x10ECECC9 returns -160 -- the public NVAPI lists 129
        // settings and none of the CUDA ones. Public ids (power management,
        // frame limiter, shader cache) accept a write from the same session.
        if (rc == -137) why = " (NVAPI_INVALID_USER_PRIVILEGE: this driver gates hidden settings behind elevation -- re-run from an elevated PowerShell)";
        if (rc == -160) why = " (NVAPI_SETTING_NOT_FOUND: this driver's NVAPI does not expose the id; try elevated, otherwise NVIDIA App / Control Panel > Manage 3D Settings is the only writer)";
        throw new Exception("NVAPI " + what + " failed: status " + rc + why);
    }

    // Returns the current value, or -1 when the setting is absent from the
    // base profile (i.e. driver default).
    public static int Get()
    {
        Check(Fn<FnVoid>(0x0150E828, "Initialize")(), "Initialize");
        IntPtr session;
        Check(Fn<FnOutPtr>(0x0694D52E, "DRS_CreateSession")(out session), "DRS_CreateSession");
        try
        {
            Check(Fn<FnPtr>(0x375DBD6B, "DRS_LoadSettings")(session), "DRS_LoadSettings");
            IntPtr profile;
            Check(Fn<FnPtrOutPtr>(0xDA8466A0, "DRS_GetBaseProfile")(session, out profile), "DRS_GetBaseProfile");
            IntPtr buf = Marshal.AllocHGlobal(SETTING_SIZE);
            try
            {
                for (int i = 0; i < SETTING_SIZE; i += 4) Marshal.WriteInt32(buf, i, 0);
                Marshal.WriteInt32(buf, 0, unchecked((int)SETTING_VER));
                int rc = Fn<FnGetSetting>(0x73BF8338, "DRS_GetSetting")(session, profile, SETTING_ID, buf);
                if (rc == NVAPI_SETTING_NOT_FOUND) return -1;
                Check(rc, "DRS_GetSetting");
                return Marshal.ReadInt32(buf, OFF_CUR);
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        finally { Fn<FnPtr>(0xDAD9CFF8, "DRS_DestroySession")(session); }
    }

    public static void Set(uint value)
    {
        Check(Fn<FnVoid>(0x0150E828, "Initialize")(), "Initialize");
        IntPtr session;
        Check(Fn<FnOutPtr>(0x0694D52E, "DRS_CreateSession")(out session), "DRS_CreateSession");
        try
        {
            Check(Fn<FnPtr>(0x375DBD6B, "DRS_LoadSettings")(session), "DRS_LoadSettings");
            IntPtr profile;
            Check(Fn<FnPtrOutPtr>(0xDA8466A0, "DRS_GetBaseProfile")(session, out profile), "DRS_GetBaseProfile");
            IntPtr buf = Marshal.AllocHGlobal(SETTING_SIZE);
            try
            {
                for (int i = 0; i < SETTING_SIZE; i += 4) Marshal.WriteInt32(buf, i, 0);
                Marshal.WriteInt32(buf, 0, unchecked((int)SETTING_VER));
                Marshal.WriteInt32(buf, OFF_ID, unchecked((int)SETTING_ID));
                Marshal.WriteInt32(buf, OFF_TYPE, 0);        // NVDRS_DWORD_TYPE
                Marshal.WriteInt32(buf, OFF_LOC, 0);         // NVDRS_CURRENT_PROFILE_LOCATION
                Marshal.WriteInt32(buf, OFF_ISCURPRE, 0);
                Marshal.WriteInt32(buf, OFF_ISPREVALID, 0);
                Marshal.WriteInt32(buf, OFF_CUR, unchecked((int)value));
                Check(Fn<FnSetSetting>(0x577DD202, "DRS_SetSetting")(session, profile, buf), "DRS_SetSetting");
                Check(Fn<FnPtr>(0xFCBC7E14, "DRS_SaveSettings")(session), "DRS_SaveSettings");
            }
            finally { Marshal.FreeHGlobal(buf); }
        }
        finally { Fn<FnPtr>(0xDAD9CFF8, "DRS_DestroySession")(session); }
    }
}
"@

if (-not ("NvDrs" -as [type])) { Add-Type -TypeDefinition $cs -Language CSharp }

function Describe([int]$v) {
    switch ($v) {
        -1 { "not set on the base profile (driver default: fallback ON for CUDA)" }
         0 { "0 = Driver Default (fallback ON for CUDA)" }
         1 { "1 = Prefer No Sysmem Fallback" }
         2 { "2 = Prefer Sysmem Fallback" }
         default { "$v = (unknown value)" }
    }
}

$before = [NvDrs]::Get()
Write-Output ("CUDA - Sysmem Fallback Policy (0x{0:X8}) on the base profile: {1}" -f [NvDrs]::SETTING_ID, (Describe $before))

if ($PSCmdlet.ParameterSetName -eq "Set") {
    if ($before -eq $Set) { Write-Output "already $Set; nothing written"; exit 0 }
    [NvDrs]::Set([uint32]$Set)
    $after = [NvDrs]::Get()
    Write-Output ("written; re-read: {0}" -f (Describe $after))
    if ($after -ne $Set) { Write-Error "re-read $after != requested $Set"; exit 1 }
    Write-Output "applies to NEW CUDA contexts only -- restart anything already holding the GPU (ollama, a WSL session with torch loaded)."
}
