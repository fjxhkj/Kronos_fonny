#NoEnv
#Warn, UseUnsetGlobal, StdOut
#Warn, LocalSameAsGlobal, StdOut
#Warn, UseUnsetLocal, Off
#Include %A_ScriptDir%
SetWorkingDir, %A_ScriptDir%
SetBatchLines, -1
ListLines Off
;~ SetKeyDelay, 20 ; 增加发送按键的稳定性

;~ 注意: 
;~ tc 传入 %P 或 %T 参数时是不包含双引号包围的路径,
;~ 如果路径中有空格,那么ahk将会识别为多个参数,
;~ 所以tc仅传递 %P 或 %T 时,必须用双引号包围.
;~ 例如:
;~ ahk_app.exe "%P" "%T"
;~ 而传入 %P%S 时则不需要使用双引号包围.
;~ 例如:
;~ ahk_app.exe %P%S

global Gs_Author := "Fonny"
, Gs_ConfigFile := path_replaceExtName(A_ScriptName, "ini")
, Gv_logFile := "_dbglog.txt"
, Gs_DebugLevel := 3
, Gv_argStr := ""
, Gv_argObj := {}

if (%0%)
{
	loop, %0%
	{
		Gv_argStr .= %A_Index% " "
		Gv_argObj.Push(%A_Index%)
	}
	Gv_argStr := RTrim(Gv_argStr)
	Gv_argObj := ArgsToObj(Gv_argObj)
}
else
{
	_useTcSelectFileAsArgs()
}

;~ 强制管理员权限运行,必须在参数解析逻辑块之后
;~ 以管理员身份运行,Vista以上将导致UAC弹窗提示
if not A_IsAdmin
{
	if (A_IsCompiled)
		Run, *RunAs "%A_ScriptFullPath%" %Gv_argStr%
	else
		Run, *RunAs "%A_AhkPath%" "%A_ScriptFullPath%" %Gv_argStr%
	ExitApp
}

main()
return

_main()
{
	if (!fsys_isByteFile(Gv_argStr))
	{
		_err := new OC_Error("传入的参数不是文件"
			, Gv_argStr
			, A_ThisFunc ;A_ThisLabel
			, A_LineFile, A_LineNumber)
		_err.Arise()
		return
	}
	
	_outFile := path_getFileNameNoExt(Gv_argStr) "_utf8.csv"
	try
	{
		if (FileExist(_outFile))
		{
			fc_FileDelete(_outFile)
		}
	}
	catch , _err
	{
		_err := new OC_Error("错误"
			, "无法删除文件: " _outFile
			, A_ThisFunc ? A_ThisFunc : A_ThisLabel
			, A_LineFile, A_LineNumber)
		_err.Arise()
		return
	}
	
	;~ FileEncoding, CP1200
	loop, Read, %Gv_argStr%, %_outFile%
	{
		if (A_Index = 1)
		{
			FileAppend, % "timestamps,open,high,low,close,volume,amount`n", % _outFile
		}
		
		_line := A_LoopReadLine
		
		;~ 格式化日期
		_line := RegExReplace(_line, "^(\d{4})\.(\d{2})\.(\d{2})", "$1-$2-$3")
		
		if (StrLen(_line) > 0)
		{
			FileAppend, % _line "`n", % _outFile
		}
	}
	Trace("Done", 3)
}

_useTcSelectFileAsArgs()
{
	_tcSelList := tc_GetSourceSelectFileFullPathList()
	Gv_argStr := Array_Splite(_tcSelList, " ")
	Gv_argObj := ArgsToObj(_tcSelList)
}

main()
{
	try
		_main()
	catch , _err
		Trace(OC_Error.Echo(_err), 1)
}

;~ #F5::Reload
;~ #F8::ExitApp
