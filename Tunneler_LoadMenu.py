# YASARA PLUGIN
# TOPIC:       Tunnels
# TITLE:       LoadCaver
# AUTHOR:      M.J.L.J. Fürst
# LICENSE:     GPL (www.gnu.org)
# DESCRIPTION: This plugin loads the menu for the tunnel inspection
#
 
"""
MainMenu: Analyze
  PullDownMenu: Tunnels
    Submenu: LoadMenu
      Request: loadmenu
"""

from yasara import *

# Yield successive n-sized chunks from lst.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def ShowButtons():
    img = MakeImage("Buttons",topcol="None",bottomcol="None")
    ShowImage(img,alpha=85,priority=1)
    PrintImage(img)
    Font("Arial",height=13,color="black")
    h = 39
    ShowButton("Tunnel on/off",x='12%', y='55%',color="White", height=h)
    ShowButton("Target on/off",x='12%', y='60%',color="White", height=h)
    ShowButton("Surf",x='3%', y='65%',color="White", height=h)
    ShowButton("H2O",x='7.5%', y='65%',color="White", height=h)
    ShowButton("SecStr",x='13%', y='65%',color="White", height=h)
    ShowButton("Nonprot",x='20%', y='65%',color="White", height=h)
    ShowButton("Color by tunnel/dist",x='12%', y='70%',color="White", height=h)
    ShowButton("Spheres",x='5%', y='75%',color="White", height=h)
    ShowButton("Balls",x='12%', y='75%',color="White", height=h)
    ShowButton("Points",x='18%', y='75%',color="White", height=h)
    ShowButton("Remove inside points",x='12%', y='80%',color="White", height=h)
    ShowButton("Exit",x='12%', y='85%',color="Red", height=h)

###########################################################################################################
###########################################################################################################

Console("off")

if request == 'loadmenu':
    if ListImage('All') != []:
        PrintImage(1)
        FillRect(color='None')
        DelImage(1)
    ShowButtons()

########################################## Button requests ###############################################
elif request == 'Tunnelonoff':
    current = SwitchObj('?Clu??????')[0]
    if current == 'Off':
        SwitchObj('?Clu??????', 'ON')
    else:
        SwitchObj('?Clu??????', 'OFF')

elif request == 'Targetonoff':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    current = SwitchObj(f'{target} {target}SS {target}H2O {target}Surf {target}NonProt')
    current = [True if x == 'On' else False for x in current]
    if any(current):
        print('some are on, so turning all off')
        print(SwitchObj(f'{target} {target}SS {target}H2O {target}Surf {target}NonProt'))
        SwitchObj(f'{target} {target}SS {target}H2O {target}Surf {target}NonProt', 'OFF')
    else:
        print('none on, so ')
        print(SwitchObj(f'{target} {target}SS {target}H2O {target}Surf {target}NonProt'))
        SwitchObj(target, 'ON')

elif request == 'SecStr':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    SwitchObj(target, 'OFF')
    if ListObj(f'{target}SS', format='OBJNUM') == []:
        new = DuplicateObj(target)[0]
        SwitchObj(new, 'ON')
        HideObj(new)
        ShowSecStrObj(new)
        NameObj(new, f'{target}SS')
    else:
        current = SwitchObj(f'{target}SS')[0]
        if current == 'Off':
            SwitchObj(f'{target}SS', 'ON')
        else:
            SwitchObj(f'{target}SS', 'OFF')

elif request == 'Nonprot':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    SwitchObj(target, 'OFF')
    if ListObj(f'{target}NonProt', format='OBJNUM') == []:
        new = DuplicateObj(target)[0]
        SwitchObj(new, 'ON')
        HideObj(new)
        HideSecStrObj(new)
        ShowRes(f'Obj {new} res !protein and !hoh')
        NameObj(new, f'{target}NonProt')
    else:
        current = SwitchObj(f'{target}NonProt')[0]
        if current == 'Off':
            SwitchObj(f'{target}NonProt', 'ON')
        else:
            SwitchObj(f'{target}NonProt', 'OFF')

elif request == 'H2O':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    SwitchObj(target, 'OFF')
    if ListObj(f'{target}H2O', format='OBJNUM') == []:
        new = DuplicateObj(target)[0]
        SwitchObj(new, 'ON')
        HideObj(new)
        HideSecStrObj(new)
        ShowRes(f'Obj {new} res hoh')
        NameObj(new, f'{target}H2O')
    else:
        current = SwitchObj(f'{target}H2O')[0]
        if current == 'Off':
            SwitchObj(f'{target}H2O', 'ON')
        else:
            SwitchObj(f'{target}H2O', 'OFF')

elif request == 'Surf':
    import re
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    current = SwitchObj(f'{target}Surf')[0]
    if current == 'Off':
        SwitchObj(f'{target}Surf', 'ON')
    else:
        SwitchObj(f'{target}Surf', 'OFF')

elif request == 'Removeinsidepoints':
    import re
    tunnel_objs = ListObj('?Clu??????')
    delete_inside =\
        ShowWin("Custom","Remove non-visible points",435,180,
                "Text",         20, 48, "Do you want to delete or just hide the points?",
                "Text",         20, 73, "Hide for a medium gain in performance",
                "Text",         20, 98, "Delete for max. performance improvement",
                "Button",      170,130,"Hide",
                "Button",      260,130,"Delete")[0]
    ShowMessage(f'Removing inside points.')
    for targetobj in tunnel_objs:
        RemoveEnvRes('all')
        AddEnvRes(targetobj)
        n_before = CountAtom(f'Obj {targetobj}')
        if delete_inside == 'Delete':
            DelAtom(f'obj {targetobj} with distance > 2.55 from accessible surface of obj {targetobj}')
        else:
            HideAtom(f'obj {targetobj} with distance > 2.55 from accessible surface of obj {targetobj}')
        n = CountAtom(f'Obj {targetobj}')
        NameObj(targetobj, re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]))
        NameObj(targetobj + 1, 
                re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]) + 'A')
        ShowMessage(f'Removed {n_before - n} inside points from object {targetobj}')
        Wait(1)
    
    HideMessage()

elif request == 'Exit':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    PrintImage(1)
    FillRect(color='None')
    DelImage(1)
    SaveSce(f'{NameObj(target)}_tunnels_exit.sce')

elif request == 'Balls':
    on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
    on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
    add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
    SwitchObj(" ".join(add_tunnels), 'on')
    SwitchObj(" ".join(on_spheres), 'off')
    BallAtom('Obj ?Clu??????')

elif request == 'Points':
    on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
    on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
    add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
    SwitchObj(" ".join(add_tunnels), 'on')
    SwitchObj(" ".join(on_spheres), 'off')
    StickAtom('Obj ?Clu??????')

elif request == 'Spheres':
    if ListObj('???_Sphere') != []:
        on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
        SwitchObj(" ".join([str(f'{x:03d}') + '_Sphere' for x in on_tunnels]), "on")
        SwitchObj('?Clu??????', 'off')
    else:
        a,r = ShowWin("Custom","Parameters for spheres",300,170,
                      "NumberInput",  20, 48,"Alpha",19,1,100,
                      "NumberInput", 175, 48,"Radius",0.45,0,1,
                      "Button",      150,120,"OK")
        tunnel_objs = " ".join([str(x) for x in ListObj('?Clu??????')])
        n = CountAtom(f'Obj {tunnel_objs}')
        if n < 3000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}')} Spheres.")
            sphere_level = 3
        elif n < 10000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}')} Spheres. This process can take several minutes.")
            sphere_level = 2
        elif n < 40000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}'):,} Spheres. This process can take very long. Click Continue or type StopPlugin in the Console")
            sphere_level = 1
            Wait('Continuebutton')
        else:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}'):,} Spheres. This process can take extremely long. Click Continue or type StopPlugin in the Console")
            sphere_level = 0
            Wait('Continuebutton')
        Wait(1)
        tunnel_objs = ListObj(tunnel_objs)


        for targetobj in tunnel_objs:
            ShowMessage(f"Creating {CountAtom(f'Obj {targetobj}'):,} Spheres of tunnel {NameObj(targetobj)[0]}")
            Wait(1)
            on_off = SwitchObj(targetobj)[0]
            SwitchObj(targetobj, 'Off')
            DelObj(f'sphere x{targetobj:03d}_sphere')
            atomlist = ListAtom(f'obj {targetobj}')
            p = PosAtom(f'obj {targetobj}',coordsys='global')
            col = ColorAtom(f'obj {targetobj}')

            modulo_value = 0
            for o in range(len(atomlist)):
                obj = ShowSphere(radius=r, color=col[o], alpha=a, level=sphere_level)
                PosObj(obj, p[(o+1)*3-3], p[(o+1)*3-2], p[(o+1)*3-1])
                if o == modulo_value:
                    modulo_value += 5000
                    jobj = ListObj('sphere')[0]
                    JoinObj('sphere', jobj)
                    ShowMessage(f'Created {o:,} / {len(atomlist):,} spheres of tunnel {NameObj(targetobj)[0]}.')
                    Wait(1)

            jobj = ListObj('sphere')[0]
            JoinObj('sphere', jobj)
            SwitchObj(jobj, on_off)
            NameObj('sphere', f'{targetobj:03d}_sphere')
    HideMessage()


elif request == 'Colorbytunneldist':
    from yasara import *

    def InsertObj(old_objnum, new_objnum):
        current_objs = ListObj('All', format='OBJNUM')
        index_start = [int(new_objnum) >= i for i in current_objs].index(False) -1
        shift_objs = current_objs[index_start:]
        [RenumberObj(i, i + 1) for i in list(reversed(shift_objs)) if i != old_objnum]
        RenumberObj(old_objnum, new_objnum)

    chunk_len=5000

    Console('Off')

    start_time = time.perf_counter()

    objs = [str(x) for x in ListObj('?Clu??????')]
    atms = [str(x) for x in ListAtom(ShowWin('AtomSelection', 'Select atom from which to calculate distance')[0])]


    min_color, max_color, per_tunnel, fast_mode = ShowWin("Custom","Parameters for tunnel colors",310,320,
                                                        "NumberInput",  20, 48,"Color of closest points",100,0,720,
                                                        "NumberInput",  20, 123,"Color of most distant points",380,0,720,
                                                        "CheckBox",     20,185,"Calculate distance per tunnel", False,
                                                        "CheckBox",     20,235,"Fast mode", True,
                                                        "Button",       150,280,"_O_K") 

    PrintCon()

    if len(atms) > 1:
        ShowMessage(f'Warning: {len(atms)} atoms match the selection for the distance calculations; using geometric mean of all selected atoms')
        Wait(35)
        c = DuplicateAtom(" ".join([str(x) for x in atms]))
        JoinObj(" ".join(str(i) for i in c), c[0])
        cx,cy,cz = PosAtom("obj " + str(c[0]), mean=True, coordsys='global')
        cen = BuildAtom("C")
        NameObj(cen, 'CenterHlp')
        PosAtom("obj " + str(cen), x = cx,y = cy, z = cz, coordsys='global')
        DelObj(c[0])
        center=str(ListAtom('Obj ' + str(cen), format='ATOMNUM')[0])
    else:
        center = " ".join(atms)
        print(f"Selected single atom {center} as center.")

    # if calculating for all points, check min and max distance first
    if not per_tunnel:
        atm_obj = ListObj('atom ' + center, format='OBJNUM')[0]
        mind = ListAtom(f'obj {" ".join(objs)} with minimum distance from atom {center}')[0]
        x = DuplicateAtom(mind)[0]
        SwapAtom('obj ' + str(x), 'Du')
        JoinObj(x, atm_obj)
        mind = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
        DelAtom(f'Obj {atm_obj} element Du')

        maxd = ListAtom(f'obj {" ".join(objs)} with maximum distance from atom {center}')[0]
        x = DuplicateAtom(maxd)[0]
        SwapAtom('obj ' + str(x), 'Du')
        JoinObj(x, atm_obj)
        maxd = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
        DelAtom(f'Obj {atm_obj} element Du')


    for tunnel in objs:
        # object=selection[0].object[j]
        # tunnel = object.number.inyas

        tname = ListObj(tunnel, format='OBJNAME')[0]
        target = ListObj(f'atom {center}', format='OBJNUM')[0]

        ShowMessage(f'Coloring tunnel {tname}')
        Wait(1)

        n = DuplicateObj(tunnel)[0]
        # Wait('continuebutton')
        natoms = CountAtom("obj " + str(n))

        SwapAtom(f'Obj {n}', "Du")
        JoinObj(n, target)
        ShowMessage(f'Obj {tunnel}: getting distances of {natoms} points')
        Wait(1)

        disto = Distance(f'obj {target} element Du', center)

        new = DuplicateAtom(f'obj {target} element Du')
        DelAtom(f'obj {target} element Du')
        NameObj(new, 'coltunnel')
        on_off = SwitchObj(tunnel)[0]
        SwitchObj(new, on_off)
        DelObj(tunnel)
        RenumberObj(n, tunnel)
        NameObj(tunnel, tname)
        SwapAtom(f'obj {tunnel}', 'H', rename=False)

        atomlist = ListAtom(f'obj {tunnel}')

        ShowMessage(f'Obj {tunnel}: Coloring {natoms:,} points')
        Wait(1)

        def rescale_floats_to_range(float_list, min_int, max_int, min_float=None, max_float=None):
            print(f'rescaling between {min_int} and {max_int}')
            if min_float == None:
                min_float = min(float_list)
            if max_float == None:
                max_float = max(float_list)
            scaled_list = [int((x - min_float) / (max_float - min_float) * (max_int - min_int) + min_int) for x in float_list]
            return scaled_list
        
        if per_tunnel:
            mind = min(disto)
            maxd = max(disto)
        
        all_cols = rescale_floats_to_range(disto, min_color, max_color, mind, maxd)

        if fast_mode:
            jmp=max(1, int(len(atomlist) / 100))
            atomlist = [x for _, x in sorted(zip(disto, atomlist))]
            all_cols = [x for _, x in sorted(zip(disto, all_cols))]
            disto = sorted(disto)
            for i, chunk in enumerate(list(chunks(atomlist, chunk_len))):
                ShowMessage(f'Obj {tunnel}: Colored {i * chunk_len:,} / {len(atomlist):,} points.')
                Wait(1)
                for j in range(0, len(chunk), jmp):
                    ColorAtom(atomlist[(i * chunk_len) + j:(i * chunk_len) + j + jmp], 
                              all_cols[(i * chunk_len) + j])
                            #   int(100 + ((disto[(i * chunk_len) + j] - mind) / (maxd - mind)) * (280)))
        else:
            for i, chunk in enumerate(list(chunks(atomlist, chunk_len))):
                ShowMessage(f'Obj {tunnel}: Colored {i * chunk_len:,} / {len(atomlist):,} points.')
                Wait(1)
                for j in range(len(chunk)):
                    ColorAtom(atomlist[(i * chunk_len) + j], int(100 + ((disto[(i * chunk_len) + j] - mind) / (maxd - mind)) * (280)))
            
        
        print(f"Done in {'{:.6f}'.format(time.perf_counter()  - start_time)} seconds")

        # StickObj(tunnel)
        HideMessage()
    DelObj('CenterHlp')


plugin.end()