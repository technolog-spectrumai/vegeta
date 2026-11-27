"""The surveillance pod for Gannet and Fulmar: it is lighter than the crate it replaces, so each drone stays up longer
than with cargo; the DRI ranges are ordered and physical (fog shortest, the horizon never passed, a fog layer seen
down through from above it); the datalink is the shorter of the horizon and the link budget; the radar does not fit
Fulmar with the gimbal; the CAD builds with the pod; the cargo numbers are unchanged."""
import pytest

import eoir_payload as ep
from fulmar import Fulmar
from gannet import Gannet


def test_pods_replace_the_crate():
    for d in ("gannet", "fulmar"):
        t = ep.pod_totals(d)
        assert 0 < t["mass_kg"] <= 5.0 and t["power_W"] > 0 and t["price_eur"] > 0
    assert ep.pod_totals("fulmar")["price_eur"] > ep.pod_totals("gannet")["price_eur"]


def test_endurance_with_the_pod():
    g, gc = Gannet.eoir_sizing(), Gannet.sizing()
    assert g["endurance_hover_min"] > gc["endurance_hover_min"] and g["endurance_loiter_min"] > g["endurance_hover_min"]
    assert g["on_station_min_at_5km"] > 10 and g["deck_hs_limit_m"] == gc["deck_hs_limit_m"] and g["wind_limit_m_s"] >= gc["wind_limit_m_s"]
    f, fc = Fulmar.eoir_sizing(), Fulmar.sizing()
    assert f["endurance_loiter_min"] > fc["endurance_min"] and f["on_station_min_at_30km"] > 30
    assert f["stall_m_s"] <= fc["stall_m_s"] and f["loiter_m_s"] >= 1.2 * f["stall_m_s"] - 0.5
    assert (f["wind_limit_m_s"], f["deck_hs_limit_m"]) == (fc["wind_limit_m_s"], fc["deck_hs_limit_m"])


@pytest.mark.parametrize("drone", ["gannet", "fulmar"])
def test_dri_is_ordered(drone):
    import sensing
    for t in ep.TARGETS:
        for alt in ep.POD[drone]["altitudes_m"]:
            r = ep.ground_dri(drone, t, alt)
            for k, v in r.items():
                assert v["detection"] >= v["recognition"] >= v["identification"] >= 0, (t, alt, k)
                assert v["detection"] <= sensing.horizon_km(alt, sensing.TARGETS[t]["height_m"]) + 0.01
            for light in ("day", "night"):
                assert r[f"{light}_fog"]["detection"] <= r[f"{light}_haze"]["detection"] <= r[f"{light}_clear"]["detection"]
        low, high = (ep.ground_dri(drone, "skiff_6m", a)["night_fog"]["detection"] for a in ep.POD[drone]["altitudes_m"])
        assert high > low                                   # above the fog layer the slant path is mostly clear


def test_fulmar_out_sees_gannet():
    g = Gannet.eoir_sizing()["dri_km"]["skiff_6m"]["120m"]["night_clear"]
    f = Fulmar.eoir_sizing()["dri_km"]["skiff_6m"]["300m"]["night_clear"]
    assert f["recognition"] > g["recognition"]               # the cooled core


def test_datalink_and_radar():
    for d in ("gannet", "fulmar"):
        for a in ep.POD[d]["altitudes_m"]:
            L = ep.link(d, a)
            assert L["range_km"] == min(L["horizon_km"], L["class_range_km"]) and L["capacity_mbps"] > L["video_mbps"]
    r = ep.radar_fits_fulmar()
    assert not r["fits_with_eoir"] and r["with_eoir_kg"] > 5.0


def test_cad_with_the_pod():
    pytest.importorskip("cadquery")
    for D in (Gannet, Fulmar):
        assert D().build(D.p({"payload": "eoir"})).val().isValid()
        assert D().build(D.p({"part": "gimbal"})).val().isValid()


def test_ladder_pod_lights_but_cannot_talk_over_outboards():
    s = Gannet.ladder_sizing()
    e = s["effectors"]
    assert s["pod"]["mass_kg"] <= 5.0 and s["endurance_hover_min"] > 15 and s["on_station_hovering_min_at_5km"] > 5
    assert e["searchlight"]["range_m"] > 400 and e["strobe"]["laser"] is False
    assert e["loudspeaker"]["intelligible_m"]["skiff_at_speed"] < 30 < e["loudspeaker"]["intelligible_m"]["skiff_idling"]
    pytest.importorskip("cadquery")
    assert Gannet().build(Gannet.p({"payload": "ladder"})).val().isValid()
